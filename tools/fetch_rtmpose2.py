"""Download an RTMPose-m body ONNX from the mirror and check it loads."""
import json, sys, urllib.request
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.fetch_birefnet import download
UA = {"User-Agent": "codex-fetch/1.0"}
repos = ["bukuroo/RTMPose-ONNX", "tasmulaev/rtmpose-m-distill", "yuvraj108c/dwpose-onnx"]
pick = None
for repo in repos:
    try:
        with urllib.request.urlopen(urllib.request.Request(
                f"https://hf-mirror.com/api/models/{repo}", headers=UA), timeout=30) as r:
            doc = json.loads(r.read().decode())
    except Exception as e:
        print(repo, "unavailable", type(e).__name__)
        continue
    files = [s["rfilename"] for s in doc.get("siblings", [])]
    print(f"\n{repo}: {len(files)} files")
    for f in files:
        print("   ", f)
    if pick is None:
        cand = [f for f in files if f.lower().endswith(".onnx") and "m" in Path(f).name.lower()]
        if not cand:
            cand = [f for f in files if f.lower().endswith(".onnx")]
        if cand:
            pick = (repo, cand[0])
if pick is None:
    raise SystemExit("no onnx found")
repo, fname = pick
url = f"https://hf-mirror.com/{repo}/resolve/main/{fname}"
dest = ROOT / "models" / Path(fname).name
print(f"\ndownloading {url} -> {dest}")
download(url, dest)
import onnxruntime as ort
so = ort.SessionOptions(); so.log_severity_level = 3
sess = ort.InferenceSession(str(dest), so, providers=["CPUExecutionProvider"])
for i in sess.get_inputs(): print("  in ", i.name, i.shape, i.type)
for o in sess.get_outputs(): print("  out", o.name, o.shape, o.type)
