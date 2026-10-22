"""Find a downloadable RTMPose-m ONNX."""
import json, sys, urllib.request, urllib.error
UA = {"User-Agent": "codex-fetch/1.0"}

def get(url, timeout=30):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, b""
    except Exception as e:
        return 0, str(e).encode()

def head(url):
    try:
        req = urllib.request.Request(url, method="HEAD", headers=UA)
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, int(r.headers.get("Content-Length") or 0), r.geturl()
    except urllib.error.HTTPError as e:
        return e.code, 0, url
    except Exception:
        return 0, -1, url

for term in ("rtmpose", "rtmdet", "dwpose"):
    st, body = get(f"https://hf-mirror.com/api/models?search={term}&limit=25")
    print(f"\n== {term}: {st}")
    if st == 200:
        for m in json.loads(body.decode()):
            print("  ", m["modelId"])
print("\n== official / github candidates")
for url in [
    "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/onnx/rtmpose-m_simcc-body7_pt-body7_420e-256x192.onnx",
    "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-m_simcc-body7_pt-body7_420e-256x192-2c1b7a9e_20230605/rtmpose-m_simcc-body7_pt-body7_420e-256x192-2c1b7a9e_20230605.onnx",
    "https://huggingface.co/hysts/rtmpose/resolve/main/rtmpose-m.onnx",
    "https://hf-mirror.com/hysts/rtmpose/resolve/main/rtmpose-m.onnx",
]:
    st, size, final = head(url)
    print(f"  {st:4d} {size/1e6:8.1f} MB {url}")
