"""Download the two public datasets into ./data (git-ignored).

BCI Competition IV 2a  -> data/bci2a/A01T.mat ... A09E.mat   (BNCI Horizon 2020 mirror)
NLB MC_Maze Medium     -> data/mc_maze_medium/*.nwb           (DANDI archive, dandiset 000139)
"""
from __future__ import annotations

import sys
from pathlib import Path

import requests

from neuralsignal.paths import DATA_DIR

BCI2A_URL = "https://bnci-horizon-2020.eu/database/data-sets/001-2014/{name}.mat"
DANDI_ASSETS = "https://api.dandiarchive.org/api/dandisets/000139/versions/draft/assets/"


def _fetch(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  have {dest.name}")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    tmp.rename(dest)
    print(f"  got  {dest.name} ({dest.stat().st_size / 1e6:.1f} MB)")


def download_bci2a(subjects=range(1, 10)) -> Path:
    out = DATA_DIR / "bci2a"
    print("BCI Competition IV 2a")
    for s in subjects:
        for sess in "TE":
            name = f"A{s:02d}{sess}"
            _fetch(BCI2A_URL.format(name=name), out / f"{name}.mat")
    return out


def download_mc_maze() -> Path:
    out = DATA_DIR / "mc_maze_medium"
    print("NLB MC_Maze Medium")
    assets = requests.get(DANDI_ASSETS, params={"page_size": 50}, timeout=60).json()["results"]
    for a in assets:
        # Only the train file carries hand velocity; the test file has spikes only.
        if "desc-train" not in a["path"]:
            continue
        url = f"https://api.dandiarchive.org/api/assets/{a['asset_id']}/download/"
        _fetch(url, out / Path(a["path"]).name)
    return out


def main(which: str = "all") -> None:
    if which in ("all", "bci2a"):
        download_bci2a()
    if which in ("all", "mc_maze"):
        download_mc_maze()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
