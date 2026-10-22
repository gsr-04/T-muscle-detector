"""Find and fetch DexiNed weights (edge detection).

Probes the official repository (GitHub API tree), the Hugging Face mirror, and a
few known re-hosts, prints what each offers, then downloads the one asked for
into models/.  Also prints the licence of the repository.

    python tools/fetch_dexined.py --list
    python tools/fetch_dexined.py --want checkpoint
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEST = ROOT / "models"
UA = {"User-Agent": "codex-fetch/1.0"}
REPOS = ["xavysp/DexiNed", "xavysp/DexiNed_PyTorch"]

# direct candidates: key -> urls (raw file or release asset)
CANDIDATES: dict[str, list[str]] = {
    "checkpoint": [
        "https://github.com/xavysp/DexiNed/raw/master/checkpoints/BIPED/10/"
        "checkpoint.pth",
        "https://raw.githubusercontent.com/xavysp/DexiNed/master/checkpoints/"
        "BIPED/10/checkpoint.pth",
        "https://github.com/xavysp/DexiNed/raw/main/checkpoints/BIPED/10/"
        "checkpoint.pth",
    ],
    "onnx": [
        "https://huggingface.co/deepghs/dexined/resolve/main/dexined.onnx",
        "https://hf-mirror.com/deepghs/dexined/resolve/main/dexined.onnx",
    ],
}


def get(url: str, timeout: int = 30) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                    timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, b""
    except Exception as e:                                   # noqa: BLE001
        return 0, str(e).encode()


def head(url: str) -> tuple[int, int, str]:
    req = urllib.request.Request(url, method="HEAD", headers=UA)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, int(r.headers.get("Content-Length") or 0), r.geturl()
    except urllib.error.HTTPError as e:
        return e.code, 0, url
    except Exception as e:                                   # noqa: BLE001
        return 0, -1, f"{type(e).__name__}: {e}"


def tree(repo: str) -> list[tuple[str, int]]:
    """Every file of the default branch, with sizes."""
    api = f"https://api.github.com/repos/{repo}/git/trees/HEAD?recursive=1"
    status, body = get(api)
    if status != 200:
        print(f"  {repo}: tree unavailable ({status})")
        return []
    doc = json.loads(body.decode("utf-8"))
    out = []
    for it in doc.get("tree", []):
        if it.get("type") == "blob":
            out.append((it["path"], int(it.get("size") or 0)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--want", default="")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--hf", default="",
                    help="list the files of these Hugging Face repos "
                         "(comma separated, via hf-mirror.com)")
    ap.add_argument("--hf-sizes", action="store_true",
                    help="with --hf: also HEAD every file for its size")
    ap.add_argument("--hf-download", default="",
                    help="namespace/name:file.onnx to fetch from hf-mirror.com")
    ap.add_argument("--hf-search", default="",
                    help="search the Hugging Face mirror for these terms")
    ap.add_argument("--gh-read", default="",
                    help="repo:path - print a file of a GitHub repo")
    ap.add_argument("--out", type=Path, default=DEST)
    ap.add_argument("--keep-repo", action="store_true",
                    help="also download the python files needed to run a .pth")
    args = ap.parse_args()

    if args.hf:
        for repo in [r.strip() for r in args.hf.split(",") if r.strip()]:
            status, body = get(f"https://hf-mirror.com/api/models/{repo}")
            if status != 200:
                print(f"  {repo}: {status}")
                continue
            doc = json.loads(body.decode("utf-8"))
            print(f"  {repo}  (license {doc.get('cardData', {}).get('license')})")
            for sib in doc.get("siblings", []):
                name = sib["rfilename"]
                if args.hf_sizes:
                    st, size, _ = head(f"https://hf-mirror.com/{repo}/resolve/"
                                       f"main/{name}")
                    txt = "?" if size <= 0 else f"{size / 1e6:8.1f} MB"
                    print(f"    {st:4d} {txt:>12s} {name}")
                else:
                    print(f"    {name}")
    if args.hf_download:
        repo, _, fname = args.hf_download.partition(":")
        url = f"https://hf-mirror.com/{repo}/resolve/main/{fname}"
        status, size, final = head(url)
        print(f"  {status} {size / 1e6:.1f} MB {url}")
        if status != 200:
            raise SystemExit("not downloadable")
        # keep the repo's own file name (external .onnx.data must match) inside
        # a folder named after the repo
        dest = args.out / repo.split("/")[-1] / fname
        from tools.fetch_birefnet import download            # noqa: WPS433
        download(url, dest)
        if dest.suffix == ".onnx":
            import onnxruntime as ort
            so = ort.SessionOptions()
            so.log_severity_level = 3
            sess = ort.InferenceSession(str(dest), so,
                                        providers=["CPUExecutionProvider"])
            for i in sess.get_inputs():
                print("   in ", i.name, i.shape, i.type)
            for o in sess.get_outputs():
                print("   out", o.name, o.shape, o.type)
        if args.list:
            return
    if args.hf_search:
        for term in [t.strip() for t in args.hf_search.split(",") if t.strip()]:
            status, body = get(f"https://hf-mirror.com/api/models"
                               f"?search={term}&limit=30")
            print(f"\n-- search {term}: {status}")
            if status != 200:
                continue
            for m in json.loads(body.decode("utf-8")):
                print(f"   {m['modelId']}")
    if args.gh_read:
        repo, _, path = args.gh_read.partition(":")
        status, body = get(f"https://raw.githubusercontent.com/{repo}/master/"
                           f"{path}")
        if status != 200:
            status, body = get(f"https://raw.githubusercontent.com/{repo}/main/"
                               f"{path}")
        print(f"\n-- {repo}/{path}: {status}")
        print(body.decode("utf-8", "replace"))

    print("== repositories")
    for repo in REPOS:
        files = tree(repo)
        if not files:
            continue
        print(f"  {repo}: {len(files)} files, "
              f"{sum(s for _, s in files) / 1e6:.1f} MB total")
        for path, size in files:
            if path.lower().endswith((".pth", ".onnx", ".pt", ".pkl")):
                print(f"    weight {path:60s} {size / 1e6:8.1f} MB")
        for path, size in files:
            if path.lower().endswith((".py", "license", "license.txt")):
                if size < 200_000:
                    print(f"    code   {path:60s} {size / 1e6:8.1f} MB")

    print("\n== hugging face mirror search")
    for host in ("https://hf-mirror.com", "https://huggingface.co"):
        status, body = get(f"{host}/api/models?search=dexined&limit=20")
        if status != 200:
            print(f"  {host}: {status} {body[:60]!r}")
            continue
        for m in json.loads(body.decode("utf-8")):
            print(f"  {host.split('//')[1]:18s} {m['modelId']}")

    print("\n== direct candidates")
    for key, urls in CANDIDATES.items():
        for url in urls:
            status, size, final = head(url)
            size_txt = "?" if size <= 0 else f"{size / 1e6:.1f} MB"
            print(f"  {key:11s} {status:4d} {size_txt:>10s} {url}")
    if args.list or not args.want:
        return

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    urls = CANDIDATES.get(args.want, [])
    for url in urls:
        status, size, final = head(url)
        if status == 200 and size > 100_000:
            dest = out_dir / (f"dexined_{args.want}" +
                              Path(url.split("?")[0]).suffix)
            from tools.fetch_birefnet import download       # noqa: WPS433
            print(f"\ndownloading {url} ({size / 1e6:.1f} MB) -> {dest}")
            download(url, dest)
            print(json.dumps({"model": str(dest)}, indent=1))
            return
    raise SystemExit(f"no downloadable candidate for {args.want}")


if __name__ == "__main__":
    main()
