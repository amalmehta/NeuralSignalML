"""Command line: `neuralsignal <command>` (or `python -m neuralsignal <command>`).

  download [all|bci2a|mc_maze]   fetch the public datasets into ./data
  train CONFIG [CONFIG ...]      run experiments (use --all for every file in configs/)
  report                         build results/Results.md and the figures in docs/figures
  spikesort                      run the synthetic spike-sorting demo
  app ...                        JSON helper used by the Mac app (see bridge.py)
"""
from __future__ import annotations

import argparse
import sys

from neuralsignal.paths import CONFIGS_DIR


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="neuralsignal", description="Neural Signal ML pipeline")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download", help="download datasets")
    d.add_argument("which", nargs="?", default="all", choices=["all", "bci2a", "mc_maze"])
    t = sub.add_parser("train", help="run experiment configs")
    t.add_argument("configs", nargs="*")
    t.add_argument("--all", action="store_true")
    sub.add_parser("report", help="aggregate results and draw figures")
    sub.add_parser("spikesort", help="synthetic spike-sorting demo")
    a = sub.add_parser("app", help="helper for the Mac app")
    a.add_argument("args", nargs=argparse.REMAINDER)
    ns = p.parse_args(argv)

    if ns.cmd == "download":
        from neuralsignal.data.download import main as dl
        dl(ns.which)
    elif ns.cmd == "train":
        from neuralsignal.experiments import run
        paths = sorted(CONFIGS_DIR.glob("*.yaml")) if ns.all else ns.configs
        if not paths:
            t.error("give config paths or --all")
        for c in paths:
            run(str(c), log=lambda m: print(m, flush=True))
    elif ns.cmd == "report":
        from neuralsignal.report import main as rep
        rep()
    elif ns.cmd == "spikesort":
        from neuralsignal.spikesort.demo import main as ss
        ss()
    elif ns.cmd == "app":
        from neuralsignal.bridge import main as br
        sys.exit(br(ns.args))


if __name__ == "__main__":
    main()
