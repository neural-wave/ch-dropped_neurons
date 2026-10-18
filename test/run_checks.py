"""Full pre-hand-in check of the submission. Run it with any Python 3 from ANY directory:

    python /full/path/to/nw26_dropped_neurons/test/run_checks.py /full/path/to/challenge_set [--seed 2026] [--work DIR]

challenge_set : the organizers' folder with energy_consumption_dev.csv, pv_labels_dev.csv, pv_extensions_labels_dev.csv.
                Omit it to run only the mock check (no real data needed).

What it does: copies src/pv_submission_combined to a scratch folder, builds a Python 3.12 venv there exactly as the
organizers would (pip install -r requirements.txt), then runs `python inference.py <csv>` (their one-argument command,
outputs in the working directory) on: a synthetic mock file, 5 random real customers, the same 5 with leading-zero ids,
the labelled extension customer POD13, and all 40 customers (md5 compared with TEAM_NOTES). Every output is validated
against the PDF format rules by test/validate.py. Nothing is written inside the repo.
"""
import argparse
import hashlib
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SUB = REPO / "src" / "pv_submission_combined"
MD5_PRED_EXPECTED = "3ea24e409d5e27f57f562a7f85587746"
MD5_EXT_EXPECTED = "99b0e71ebf7e631849738e3efed020d5"


def say(msg):
    print(f"\n== {msg}", flush=True)


def die(msg):
    print(f"\nFAILED: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def run(cmd, cwd=None, capture=False):
    r = subprocess.run([str(c) for c in cmd], cwd=cwd, text=True, capture_output=capture)
    if r.returncode != 0:
        if capture:
            print(r.stdout); print(r.stderr, file=sys.stderr)
        die(f"command failed ({r.returncode}): {' '.join(str(c) for c in cmd)}")
    return r


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def find_python312():
    if sys.version_info[:2] == (3, 12):
        return sys.executable
    for name in ("python3.12", "python3.12.exe"):
        if shutil.which(name):
            return shutil.which(name)
    if platform.system() == "Windows" and shutil.which("py"):
        r = subprocess.run(["py", "-3.12", "-c", "import sys; print(sys.executable)"], capture_output=True, text=True)
        if r.returncode == 0:
            return r.stdout.strip()
    die("Python 3.12 not found (macOS: brew install python@3.12, Ubuntu: apt install python3.12 python3.12-venv, "
        "Windows: python.org installer). Or run this script with a 3.12 interpreter.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("challenge_set", nargs="?", help="organizers' data folder; omit for the mock check only")
    ap.add_argument("--seed", type=int, default=2026, help="seed of the 5-customer sample (2026 = verified reference)")
    ap.add_argument("--work", default=str(Path(tempfile.gettempdir()) / "pvtest"), help="scratch folder (default: <tmp>/pvtest)")
    a = ap.parse_args()

    data = Path(a.challenge_set).resolve() if a.challenge_set else None
    work = Path(a.work).resolve()
    if not (SUB / "inference.py").is_file():
        die(f"submission not found at {SUB}")
    if work == Path(work.anchor) or work in REPO.parents or work == REPO:
        die(f"unsafe work folder {work}")
    if data:
        for f in ("energy_consumption_dev.csv", "pv_labels_dev.csv", "pv_extensions_labels_dev.csv"):
            if not (data / f).is_file():
                die(f"missing {data / f}")
    py312 = find_python312()

    say("0. settings")
    print(f"repo:  {REPO}\ndata:  {data or '<none, mock only>'}\nwork:  {work}\nseed:  {a.seed}\npy312: {py312}")

    say("1. copy the submission and build its venv the organizers' way")
    if work.exists():
        shutil.rmtree(work)
    sub = work / "submission"
    shutil.copytree(SUB, sub, ignore=shutil.ignore_patterns(".venv*", "__pycache__", "predictions.csv", "pv_extensions_predictions.csv"))
    run([py312, "-m", "venv", sub / ".venv"])
    vpy = sub / ".venv" / ("Scripts/python.exe" if platform.system() == "Windows" else "bin/python")
    run([vpy, "-m", "pip", "install", "-q", "--upgrade", "pip"])
    if platform.system() == "Linux":
        run([vpy, "-m", "pip", "install", "-q", "torch==2.14.1", "--index-url", "https://download.pytorch.org/whl/cpu"])
    run([vpy, "-m", "pip", "install", "-q", "-r", sub / "requirements.txt"])
    run([vpy, "-c", "import torch, sklearn, pandas, pyarrow; print('versions:', torch.__version__, sklearn.__version__, pandas.__version__, pyarrow.__version__)"])
    inf, val = sub / "inference.py", REPO / "test" / "validate.py"

    def infer_and_validate(name, csv, labels, ext_labels):
        """Organizers' one-argument command, run inside an empty folder so the outputs land in its CWD."""
        out = work / f"run_{name}"
        out.mkdir(parents=True)
        r = subprocess.run([str(vpy), str(inf), str(csv)], cwd=out, text=True, capture_output=True)
        print(r.stdout.strip())
        if r.returncode != 0 or r.stderr.strip():
            print(r.stderr, file=sys.stderr)
            die(f"{name}: inference exited {r.returncode} or wrote to stderr (a fallback fired)")
        for f in ("predictions.csv", "pv_extensions_predictions.csv"):
            print(f"--- {out / f}"); print((out / f).read_text(encoding="utf-8").rstrip())
        v = subprocess.run([str(vpy), str(val), str(csv), str(out), str(labels), str(ext_labels)], text=True, capture_output=True)
        print("\n".join(l for l in v.stdout.splitlines() if "FAIL" in l or "TP=" in l or l.startswith("RESULT")))
        if "ALL FORMAT CHECKS PASSED" not in v.stdout:
            print(v.stdout); print(v.stderr, file=sys.stderr)
            die(f"{name}: format validation failed")
        return out

    say("2. mock input in the organizers' schema (synthetic, no real data)")
    mock = work / "mock"
    run([vpy, REPO / "test" / "make_mock.py", mock])
    infer_and_validate("mock", mock / "test_set.csv", mock / "mock_labels.csv", mock / "mock_extensions_labels.csv")

    if data is None:
        say("done (mock only). Pass the challenge_set folder as first argument for the real-data checks.")
        return
    ts, lab, ext = data / "energy_consumption_dev.csv", data / "pv_labels_dev.csv", data / "pv_extensions_labels_dev.csv"

    say(f"3. five random real customers (seed {a.seed}), organizers' one-argument command")
    s5 = work / "test_set_5.csv"
    run([vpy, REPO / "test" / "make_sample.py", ts, s5, a.seed])
    out5 = infer_and_validate("sample5", s5, lab, ext)

    say("4. same five with leading-zero ids (ids must be preserved as strings)")
    ids, lines = {}, []
    with open(s5, encoding="utf-8") as f:
        lines.append(f.readline())
        for line in f:
            cid, rest = line.split(";", 1)
            ids.setdefault(cid, "0" * (3 + len(ids)) + str(len(ids) + 1))
            lines.append(ids[cid] + ";" + rest)
    sz = work / "test_set_5_zeros.csv"
    sz.write_text("".join(lines), encoding="utf-8")
    print("id mapping:", ids)
    outz = work / "run_zeros"
    outz.mkdir()
    r = subprocess.run([str(vpy), str(inf), str(sz)], cwd=outz, text=True, capture_output=True)
    print(r.stdout.strip())
    if r.returncode != 0 or r.stderr.strip():
        die(f"zeros: inference failed\n{r.stderr}")
    pz = (outz / "predictions.csv").read_text(encoding="utf-8").splitlines()
    p5 = (out5 / "predictions.csv").read_text(encoding="utf-8").splitlines()
    print("\n".join(pz))
    if [l.split(";", 1)[1] for l in pz] != [l.split(";", 1)[1] for l in p5]:
        die("zeros: predictions differ between POD ids and zero-padded ids")
    if not all(l.split(";", 1)[0].startswith("0") for l in pz[1:]):
        die(f"zeros: leading zeros lost: {[l.split(';', 1)[0] for l in pz[1:]]}")
    print("leading zeros preserved, predictions identical")

    say("5. the labelled extension customer POD13")
    s13 = work / "test_set_pod13.csv"
    with open(ts, encoding="utf-8") as f, open(s13, "w", encoding="utf-8") as g:
        g.write(f.readline())
        for line in f:
            if line.startswith("POD13;"):
                g.write(line)
    infer_and_validate("pod13", s13, lab, ext)

    say("6. all 40 customers, md5 against TEAM_NOTES")
    out40 = infer_and_validate("all40", ts, lab, ext)
    p, e = md5(out40 / "predictions.csv"), md5(out40 / "pv_extensions_predictions.csv")
    print(f"predictions.csv md5 {p} (expected {MD5_PRED_EXPECTED})\nextensions      md5 {e} (expected {MD5_EXT_EXPECTED})")
    if (p, e) != (MD5_PRED_EXPECTED, MD5_EXT_EXPECTED):
        die("md5 mismatch: the pipeline output changed vs TEAM_NOTES")

    say("7. hand-in folder clean?")
    stray = [q.name for q in SUB.iterdir()
             if q.name.startswith(".venv") or q.name == "__pycache__" or q.name in ("predictions.csv", "pv_extensions_predictions.csv")]
    stray += [str(q.relative_to(SUB)) for q in SUB.rglob("__pycache__") if q.parent != SUB]
    if stray:
        print(f"WARNING: remove these from {SUB} before zipping:\n  " + "\n  ".join(stray))
    else:
        print("clean")

    say(f"ALL CHECKS PASSED   (runs kept in {work}; delete that folder when done)")


if __name__ == "__main__":
    main()
