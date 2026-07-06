"""
Preview utility — runs any animation function in preview mode.

Instead of rendering a full .mp4, patches FuncAnimation.save() to capture
5 key frames (0%, 25%, 50%, 75%, 100%) and saves them as PNGs to preview/.

Usage:
    python main.py animate ... --preview

Works by monkey-patching matplotlib.animation.FuncAnimation before the
animation callable runs, then restoring it. No changes to animation
functions required.
"""
import os
import matplotlib.animation as _anim
import matplotlib.pyplot as plt


class _PreviewCapture:
    """Drop-in replacement for FuncAnimation that saves key frames as PNGs."""

    def __init__(self, fig, update_fn, frames=None, **kwargs):
        self.fig = fig
        self.update_fn = update_fn
        n = frames if isinstance(frames, int) else len(list(frames))
        self._frames = list(range(n))

    def save(self, out_path, **kwargs):
        os.makedirs("preview", exist_ok=True)
        stem = os.path.splitext(os.path.basename(out_path))[0]
        n = len(self._frames)
        key_indices = sorted({0, n // 4, n // 2, 3 * n // 4, n - 1})
        for idx in key_indices:
            frame = self._frames[idx]
            self.update_fn(frame)
            pct = int(idx / max(n - 1, 1) * 100)
            png_path = f"preview/{stem}_{pct:03d}pct.png"
            self.fig.savefig(png_path, dpi=120, bbox_inches="tight")
            print(f"  Preview: {png_path}")


def run_preview(animate_callable):
    """
    Run animate_callable() with FuncAnimation replaced by _PreviewCapture.
    Saves 5 key-frame PNGs to preview/ instead of rendering a video.
    """
    original = _anim.FuncAnimation
    _anim.FuncAnimation = _PreviewCapture
    try:
        animate_callable()
    finally:
        _anim.FuncAnimation = original
        plt.close("all")
