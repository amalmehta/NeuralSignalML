"""Draw the app icon (stacked EEG-like traces on a dark rounded square) -> app/Resources/AppIcon.icns."""
import subprocess
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

out = Path(__file__).parent / "Resources"
fig = plt.figure(figsize=(10.24, 10.24), dpi=100)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
fig.patch.set_alpha(0)
ax.add_patch(FancyBboxPatch((0.1, 0.1), 0.8, 0.8, boxstyle="round,pad=0,rounding_size=0.18",
                            fc="#14182b", ec="none"))
rng = np.random.default_rng(3)
t = np.linspace(0.17, 0.83, 600)
colors = ["#4C78A8", "#72B7B2", "#F58518", "#E45756"]
for k, (y0, c) in enumerate(zip([0.68, 0.56, 0.44, 0.32], colors)):
    burst = np.exp(-((t - 0.5 - 0.06 * (k - 1.5)) / 0.08) ** 2)
    sig = 0.035 * np.sin(2 * np.pi * (11 + 2 * k) * t * 3) * (0.35 + burst) + 0.006 * rng.standard_normal(t.size)
    ax.plot(t, y0 + sig, color=c, lw=9, solid_capstyle="round")
with tempfile.TemporaryDirectory() as d:
    png = Path(d) / "icon.png"
    fig.savefig(png, transparent=True)
    iconset = Path(d) / "AppIcon.iconset"; iconset.mkdir()
    for s in (16, 32, 64, 128, 256, 512):
        for scale in (1, 2):
            subprocess.run(["sips", "-z", str(s * scale), str(s * scale), str(png), "--out",
                            str(iconset / f"icon_{s}x{s}{'@2x' if scale == 2 else ''}.png")],
                           check=True, capture_output=True)
    subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(out / "AppIcon.icns")], check=True)
    import shutil; shutil.copy(png, out / "AppIcon.png")
print("wrote", out / "AppIcon.icns")
