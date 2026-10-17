# --- Colab cell 1 (no restart needed) ---
# !pip install -q gradio

# --- Colab cell 2: gradio labeling tool, click the image to set the PV start day ---
import os, gc, time, functools, numpy as np, pandas as pd, matplotlib.pyplot as plt, gradio as gr
import pyarrow as pa, pyarrow.csv as pv, pyarrow.compute as pc

data_dir = "./AIL_data_for_neural_wave_v1.1/challenge_set/"
out = "manual_labels.csv"

cache = "unlabelled_pod.parquet"  # fast copy of the prepared data, survives runtime restarts
if "df" in globals():
    print("data already in memory, not reloading")
elif os.path.exists(cache):
    print("loading cached data..."); df = pd.read_parquet(cache)
else:
    print("loading data (first time only)...")
    # read with pyarrow: timestamps parsed natively (no python strings), POD filter before pandas
    tbl = pv.read_csv(data_dir + "energy_consumption_unlabelled_dev.csv", parse_options=pv.ParseOptions(delimiter=";"),
                      convert_options=pv.ConvertOptions(include_columns=["customer_id", "energy_consumption_kwh", "timestamp_local"],
                                                        column_types={"timestamp_local": pa.timestamp("ms"), "energy_consumption_kwh": pa.float32()}))
    tbl = tbl.filter(pc.starts_with(tbl["customer_id"], "POD"))  # empty ids are dropped too
    df = tbl.to_pandas(strings_to_categorical=True, self_destruct=True, split_blocks=True); del tbl  # frees arrow memory while converting
    t = df.pop("timestamp_local") - pd.Timedelta("15min")  # timestamps mark interval end
    df["day"], df["step"] = t.dt.normalize(), (t.dt.hour * 4 + t.dt.minute // 15).astype("int8")
    del t; gc.collect(); pa.default_memory_pool().release_unused()  # give freed memory back to the system
    df.to_parquet(cache)
print("loaded", df.customer_id.nunique(), "customers,", round(df.memory_usage(deep=True).sum() / 1e9, 2), "GB")

cols = ["customer_id", "pv_installed", "pv_installation_date", "confidence"]
def read_labels():  # tolerant: old 3-column rows get an empty confidence
    return pd.read_csv(out, names=cols, skiprows=1) if os.path.exists(out) else pd.DataFrame(columns=cols)
if os.path.exists(out): read_labels().to_csv(out, index=False)  # repair the file once to the 4-column layout

queue, s = [], {}
rng = np.random.default_rng(42)  # fixed seed: same batches every run, still different on each pull

def timed(f):  # log how long each button/click takes, to tell python time from Colab-proxy delay
    @functools.wraps(f)
    def wrapper(*a, **kw):
        t0 = time.time(); r = f(*a, **kw)
        with open("timing.log", "a") as fh: fh.write(f"{time.strftime('%X')} {f.__name__} {time.time() - t0:.3f}s\n")
        return r
    return wrapper

def to_rgb(z):  # values in [0, 1] -> uint8 RGB with the viridis colour map
    return (plt.cm.viridis(z)[..., :3] * 255).astype(np.uint8)

def draw():  # image as RGB array: x = day, y = timestep (midnight at bottom)
    if not queue: return None, "**Batch done.** Pull different 10?"
    img, start = s["img"], s["start"]
    m = img.values.max() + 1e-9                          # colour range: 0 (dark purple) .. m (yellow)
    s["k"] = k = max(1, 900 // img.shape[1])             # widen short series to ~900 px

    pic = to_rgb(img.values[::-1] / m)                   # flip rows so midnight is at the bottom
    pic = pic.repeat(4, axis=0).repeat(k, axis=1)        # enlarge: 4 px per timestep, k px per day
    dash = (np.arange(pic.shape[0]) // 8) % 2 == 0                         # 8 px on, 8 px off
    for d in np.flatnonzero(img.columns.year[1:] != img.columns.year[:-1]) + 1:  # first day of each new year
        pic[dash, d * k: d * k + 2] = 255                                    # white dashed year separator
    if start is not None: pic[:, start * k: start * k + max(k, 4)] = [255, 0, 0]  # red PV start line

    bar = to_rgb(np.linspace(0, 1, pic.shape[1]))[None].repeat(20, axis=0)  # colour legend
    gap = np.full((10, pic.shape[1], 3), 255, np.uint8)
    pic = np.vstack([pic, gap, bar])

    picked = img.columns[start].date() if start is not None else "no click = day 0"
    info = (f"**{queue[0]}** ({len(queue)} left) · days {img.columns[0].date()} → {img.columns[-1].date()} · **PV start: {picked}**"
            f"  \ncolour bar (bottom): 0 kWh (left) → {m:.1f} kWh (right)")
    return pic, info

def load():  # next customer from the queue
    if queue:
        s["img"] = df[df.customer_id == queue[0]].pivot_table(index="step", columns="day", values="energy_consumption_kwh").fillna(0)
    s["start"] = None
    return draw()

@timed
def new_batch():  # 10 random customers not labeled yet
    done = set(read_labels().customer_id)
    todo = pd.Series(df.customer_id.unique()).loc[lambda x: ~x.isin(done)]
    queue[:] = list(todo.sample(min(10, len(todo)), random_state=rng))
    return load()

@timed
def click(evt: gr.SelectData):  # x pixel -> day index
    if queue: s["start"] = min(evt.index[0] // s["k"], s["img"].shape[1] - 1)
    return draw()

@timed
def save(pv, conf):  # append one row to the csv, go to next
    if not queue: return draw()
    if pv and s["start"] is None: s["start"] = 0  # no click = PV from day 0
    pd.DataFrame([{"customer_id": queue[0], "pv_installed": pv,
                   "pv_installation_date": s["img"].columns[s["start"]].date() if pv else None,
                   "confidence": conf}]
                 ).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
    queue.pop(0)
    return load()

@timed
def back():  # undo last saved label: drop it from the csv and show that customer again
    labs = read_labels()
    if labs.empty: return draw()
    queue.insert(0, labs.customer_id.iloc[-1])
    labs.iloc[:-1].to_csv(out, index=False)
    return load()

with gr.Blocks() as demo:
    info = gr.Markdown()
    image = gr.Image(label="x = day, y = timestep of day", interactive=False)
    conf = gr.Slider(0, 100, value=100, step=10, label="How confident are you? (%)")
    with gr.Row():
        b_back = gr.Button("⬅ Back (undo last)")
        b_save, b_nopv, b_new = gr.Button("Save & next", variant="primary"), gr.Button("No PV & next"), gr.Button("Pull different 10")
    image.select(click, None, [image, info])
    b_save.click(lambda c: save(True, c), conf, [image, info])
    b_nopv.click(lambda c: save(False, c), conf, [image, info])
    b_new.click(new_batch, None, [image, info])
    b_back.click(back, None, [image, info])
    demo.load(new_batch, None, [image, info])

from google.colab import output
port = 7860
url = output.eval_js(f"google.colab.kernel.proxyPort({port})")  # Colab's private https address for this port
gr.close_all()  # free the port if the cell is rerun
demo.launch(share=False, inline=False, prevent_thread_lock=True, show_error=True,  # no public link, errors shown in the app
            server_port=port, root_path=url.rstrip("/"))  # make image links use the proxy address
output.serve_kernel_port_as_iframe(port, height=900)
