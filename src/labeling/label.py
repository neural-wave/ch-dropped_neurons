# --- Colab cell 1 (no restart needed) ---
# !pip install -q gradio

# --- Colab cell 2: gradio labeling tool, click the image to set the PV start day ---
import os, gc, numpy as np, pandas as pd, matplotlib.pyplot as plt, gradio as gr
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

def draw():  # image as RGB array: x = day, y = timestep (midnight at bottom)
    if not queue: return None, "**Batch done.** Pull different 10?"
    img = s["img"]; mean = img.values.mean(); m = np.abs(img.values - mean).max() + 1e-9
    z = (img.values - mean) / m / 2 + 0.5  # mean -> 0.5 (white), below -> blue, above -> red
    s["k"] = k = max(1, 900 // z.shape[1])  # widen short series
    rgb = (plt.cm.RdBu_r(z)[::-1, :, :3] * 255).astype(np.uint8).repeat(4, axis=0).repeat(k, axis=1)
    if s["start"] is not None: rgb[:, s["start"] * k: s["start"] * k + max(k, 4)] = [0, 0, 0]  # black: red is now part of the colour scale
    bar = (plt.cm.RdBu_r(np.linspace(0, 1, rgb.shape[1]))[None, :, :3] * 255).astype(np.uint8).repeat(20, axis=0)
    rgb = np.vstack([rgb, np.full((10, rgb.shape[1], 3), 255, np.uint8), bar])  # colour legend at the bottom
    picked = img.columns[s["start"]].date() if s["start"] is not None else "no click = day 0"
    return rgb, (f"**{queue[0]}** ({len(queue)} left) · days {img.columns[0].date()} → {img.columns[-1].date()} · **PV start: {picked}**"
                 f"  \ncolour bar (bottom): blue {mean - m:.1f} kWh → white = mean {mean:.1f} kWh → red {mean + m:.1f} kWh")

def load():  # next customer from the queue
    if queue:
        s["img"] = df[df.customer_id == queue[0]].pivot_table(index="step", columns="day", values="energy_consumption_kwh").fillna(0)
    s["start"] = None
    return draw()

def new_batch():  # 10 random customers not labeled yet
    done = set(read_labels().customer_id)
    todo = pd.Series(df.customer_id.unique()).loc[lambda x: ~x.isin(done)]
    queue[:] = list(todo.sample(min(10, len(todo))))
    return load()

def click(evt: gr.SelectData):  # x pixel -> day index
    if queue: s["start"] = min(evt.index[0] // s["k"], s["img"].shape[1] - 1)
    return draw()

def save(pv, conf):  # append one row to the csv, go to next
    if not queue: return draw()
    if pv and s["start"] is None: s["start"] = 0  # no click = PV from day 0
    pd.DataFrame([{"customer_id": queue[0], "pv_installed": pv,
                   "pv_installation_date": s["img"].columns[s["start"]].date() if pv else None,
                   "confidence": conf}]
                 ).to_csv(out, mode="a", header=not os.path.exists(out), index=False)
    queue.pop(0)
    return load()

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
