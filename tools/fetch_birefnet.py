"""Fetch a BiRefNet ONNX model into models/ and check it loads.

Probes the known mirrors first (rembg's GitHub release assets and the ONNX
exports on the Hugging Face Hub), prints what is there, then downloads the one
asked for.  Nothing else in the project is touched.

    python tools/fetch_birefnet.py --list
    python tools/fetch_birefnet.py --want lite
    python tools/fetch_birefnet.py --want general
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "models"

CANDIDATES: dict[str, list[str]] = {
    "lite": [
        "https://huggingface.co/onnx-community/BiRefNet_lite-ONNX/resolve/main/"
        "onnx/model.onnx",
        "https://huggingface.co/onnx-community/BiRefNet_lite-ONNX/resolve/main/"
        "onnx/model_quantized.onnx",
        "https://github.com/danielgatis/rembg/releases/download/v0.0.0/"
        "birefnet-general-lite.onnx",
    ],
    "general": [
        "https://github.com/danielgatis/rembg/releases/download/v0.0.0/"
        "birefnet-general.onnx",
        "https://huggingface.co/onnx-community/BiRefNet-ONNX/resolve/main/"
        "onnx/model.onnx",
    ],
    "portrait": [
        "https://github.com/danielgatis/rembg/releases/download/v0.0.0/"
        "birefnet-portrait.onnx",
    ],
    "isnet": [
        "https://github.com/danielgatis/rembg/releases/download/v0.0.0/"
        "isnet-general-use.onnx",
    ],
}
UA = {"User-Agent": "codex-fetch/1.0"}

# the same files behind the mirrors that are usually reachable from here
MIRRORS = ["https://hf-mirror.com/", "https://huggingface.co/"]


def expand(url: str) -> list[str]:
    if "huggingface.co/" not in url:
        return [url]
    tail = url.split("huggingface.co/", 1)[1]
    return [m + tail for m in MIRRORS]


def dest_name(key: str) -> str:
    """birefnet_<key>.onnx - the mirror renames the file to its blob hash."""
    return f"birefnet_{key}.onnx"


def gh_assets() -> None:
    """List the ONNX assets of rembg's release, whose names we have to know."""
    api = ("https://api.github.com/repos/danielgatis/rembg/releases"
           "?per_page=10")
    try:
        with urllib.request.urlopen(urllib.request.Request(api, headers=UA),
                                    timeout=30) as r:
            rel = json.loads(r.read().decode("utf-8"))
    except Exception as e:                                   # noqa: BLE001
        print(f"github api unavailable: {type(e).__name__}: {e}")
        return
    for rel_i in rel:
        print(f"  release {rel_i.get('tag_name')}:")
        for a in rel_i.get("assets", []):
            print(f"    {a['name']:38s} {a['size'] / 1e6:8.1f} MB")


def head(url: str) -> tuple[int, int, str]:
    """Status, size and the URL after redirects (0/-1/- when unreachable)."""
    req = urllib.request.Request(url, method="HEAD", headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            size = int(r.headers.get("Content-Length") or 0)
            return r.status, size, r.geturl()
    except urllib.error.HTTPError as e:
        return e.code, 0, url
    except Exception as e:                                   # noqa: BLE001
        return 0, -1, f"{type(e).__name__}: {e}"


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    headers = dict(UA)
    if have:
        headers["Range"] = f"bytes={have}-"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=600) as r, \
            part.open("ab" if have else "wb") as fh:
        total = int(r.headers.get("Content-Length") or 0)
        if have and r.status == 206:
            total += have
            print(f"    resuming at {have / 1e6:.1f} MB")
        else:
            have = 0
        got = have
        step = 64 * 1024 * 1024
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
            got += len(chunk)
            if total and got // step != (got - len(chunk)) // step:
                print(f"    {got / 1e6:7.1f} / {total / 1e6:.1f} MB",
                      flush=True)
    part.replace(dest)
    print(f"    wrote {dest} ({dest.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--want", default="lite,general",
                    help="comma separated: " + ", ".join(sorted(CANDIDATES)))
    ap.add_argument("--list", action="store_true", help="only probe the mirrors")
    ap.add_argument("--out", type=Path, default=DEST)
    args = ap.parse_args()

    want = [w.strip() for w in args.want.split(",") if w.strip()]
    if args.list:
        gh_assets()
    found: dict[str, tuple[str, int, str]] = {}
    for key in want:
        for raw in CANDIDATES.get(key, []):
            for url in expand(raw):
                status, size, final = head(url)
                size_txt = "?" if size <= 0 else f"{size / 1e6:.1f} MB"
                print(f"{key:9s} {status:4d} {size_txt:>10s} {url}")
                if (status == 200 and size > 1_000_000
                        and key not in found):
                    found[key] = (url, size, final)
    if args.list:
        return
    if not found:
        raise SystemExit("nothing downloadable found")
    dests = []
    for key, (url, size, final) in found.items():
        name = dest_name(key)
        dest = args.out / name
        print(f"\ndownloading {key} ({size / 1e6:.1f} MB) -> {dest}")
        if not dest.exists() or dest.stat().st_size != size:
            download(url, dest)
        else:
            print("    already there, size matches")
        dests.append(dest)

    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    for dest in dests:
        sess = ort.InferenceSession(str(dest), so,
                                    providers=["CPUExecutionProvider"])
        print(f"\n{dest.name}")
        for i in sess.get_inputs():
            print("   in ", i.name, i.shape, i.type)
        for o in sess.get_outputs():
            print("   out", o.name, o.shape, o.type)
    print(json.dumps({"models": [str(d) for d in dests]}, indent=1))


if __name__ == "__main__":
    main()
