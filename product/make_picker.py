"""Per-person picker prototype (plan §Product surfaces).

Generates a self-contained static HTML viewer: filmstrip of candidate donor
frames per person, swipe / arrow-key to step frame by frame, press-and-hold
to compare against the original, and the synthesis level labeled per frame
(A / B / B+fill / rejected). No runtime dependencies — open index.html.

Usage:  PYTHONPATH=.. python3 product/make_picker.py [--out product/picker_out]
"""
import argparse
import base64
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from besttake.adapters.synthetic_capture import SyntheticCapture
from besttake.common.types import FaceObservation
from besttake.face_model.canonical import SyntheticHeadModel
from besttake.level_b.pipeline import DonorInput, LevelBConfig, LevelBSwap

HTML = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Best Take — pick the frame</title>
<style>
  :root { color-scheme: dark; }
  body { font-family: -apple-system, system-ui, sans-serif; background: #111;
         color: #eee; margin: 0; display: flex; flex-direction: column;
         align-items: center; min-height: 100vh; }
  h1 { font-size: 17px; font-weight: 600; margin: 18px 0 4px; }
  .sub { color: #999; font-size: 12.5px; margin-bottom: 14px; }
  .stage { position: relative; width: min(92vw, 420px); aspect-ratio: 1;
           border-radius: 18px; overflow: hidden; background: #000;
           touch-action: pan-y; cursor: grab; }
  .stage img { position: absolute; inset: 0; width: 100%; height: 100%;
               object-fit: contain; -webkit-user-drag: none; user-select: none; }
  .stage .orig { opacity: 0; transition: opacity 120ms; }
  .stage.hold .orig { opacity: 1; }
  .badge { position: absolute; left: 10px; top: 10px; background: #000a;
           backdrop-filter: blur(6px); padding: 4px 10px; border-radius: 999px;
           font-size: 12px; letter-spacing: .4px; }
  .badge.rejected { background: #a32020cc; }
  .hint { position: absolute; right: 10px; bottom: 10px; color: #fffc;
          font-size: 11px; background: #000a; padding: 3px 8px;
          border-radius: 999px; }
  .strip { display: flex; gap: 8px; margin: 16px 0 6px; overflow-x: auto;
           padding: 6px 12px; max-width: 94vw; scroll-behavior: smooth; }
  .thumb { flex: 0 0 auto; width: 64px; height: 64px; border-radius: 10px;
           overflow: hidden; border: 2px solid #333; opacity: .55;
           position: relative; }
  .thumb img { width: 100%; height: 100%; object-fit: cover; }
  .thumb.sel { border-color: #4da3ff; opacity: 1; }
  .thumb .lv { position: absolute; left: 3px; bottom: 3px; font-size: 9px;
               background: #000b; padding: 1px 4px; border-radius: 5px; }
  .person { display: flex; gap: 8px; margin-top: 4px; }
  .person button { background: #222; color: #eee; border: 1px solid #333;
                   border-radius: 999px; padding: 6px 14px; font-size: 13px; }
  .person button.sel { background: #2b4b6f; border-color: #4da3ff; }
  .nav { display: flex; gap: 14px; margin: 10px 0 24px; }
  .nav button { font-size: 20px; width: 52px; height: 40px; border-radius: 12px;
                background: #222; color: #eee; border: 1px solid #333; }
</style></head><body>
<h1>Best take — swipe the burst</h1>
<div class="sub">◀ ▶ keys or swipe the photo · press &amp; hold to compare ·
badge shows the synthesis level used</div>
<div class="stage" id="stage">
  <img id="imgSwap" alt="best-take result">
  <img id="imgOrig" class="orig" alt="original frame">
  <div class="badge" id="badge">A</div>
  <div class="hint">hold = original</div>
</div>
<div class="strip" id="strip"></div>
<div class="person" id="persons"></div>
<div class="nav">
  <button id="prev" aria-label="previous frame">‹</button>
  <button id="next" aria-label="next frame">›</button>
</div>
<script>
const DATA = __DATA__;
let person = 0, idx = 0;
const stage = document.getElementById('stage'),
      imgSwap = document.getElementById('imgSwap'),
      imgOrig = document.getElementById('imgOrig'),
      badge = document.getElementById('badge'),
      strip = document.getElementById('strip');

function render() {
  const frames = DATA[person].frames;
  const f = frames[idx];
  imgSwap.src = f.result; imgOrig.src = f.original;
  badge.textContent = f.level + (f.coverage ? ' · ' + Math.round(100*f.coverage) + '% swapped' : '');
  badge.className = 'badge' + (f.level === 'rejected' ? ' rejected' : '');
  document.querySelectorAll('.thumb').forEach((t, i) => {
    t.classList.toggle('sel', i === idx);
    if (i === idx) t.scrollIntoView({inline: 'center', block: 'nearest'});
  });
}
function step(d) { idx = (idx + d + DATA[person].frames.length) % DATA[person].frames.length; render(); }

function buildStrip() {
  strip.innerHTML = '';
  DATA[person].frames.forEach((f, i) => {
    const d = document.createElement('div');
    d.className = 'thumb';
    d.innerHTML = `<img src="${f.result}"><span class="lv">${f.level}</span>`;
    d.onclick = () => { idx = i; render(); };
    strip.appendChild(d);
  });
}
function buildPersons() {
  const box = document.getElementById('persons');
  box.innerHTML = '';
  DATA.forEach((p, i) => {
    const b = document.createElement('button');
    b.textContent = p.name;
    b.className = i === person ? 'sel' : '';
    b.onclick = () => { person = i; idx = 0; buildStrip(); render(); buildPersons(); };
    box.appendChild(b);
  });
}
document.getElementById('prev').onclick = () => step(-1);
document.getElementById('next').onclick = () => step(1);
addEventListener('keydown', e => {
  if (e.key === 'ArrowLeft') step(-1);
  if (e.key === 'ArrowRight') step(1);
});
stage.addEventListener('pointerdown', () => stage.classList.add('hold'));
addEventListener('pointerup', () => stage.classList.remove('hold'));
addEventListener('pointercancel', () => stage.classList.remove('hold'));
// swipe left/right on the stage
let x0 = null;
stage.addEventListener('pointerdown', e => x0 = e.clientX);
stage.addEventListener('pointerup', e => {
  if (x0 === null) return;
  const dx = e.clientX - x0;
  if (Math.abs(dx) > 40) step(dx < 0 ? 1 : -1);
  x0 = null;
});
buildStrip(); buildPersons(); render();
</script></body></html>
"""


def _to_data_uri(img: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(
        buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def build(model: SyntheticHeadModel, out_dir: Path, n_frames: int = 6) -> Path:
    """One capture, one person; every donor frame becomes a swipeable
    candidate with its own Level B result."""
    rng = np.random.default_rng(7)
    cid = np.array([0.3, -0.4, 0.2, 0.1, -0.2, 0.3, 0.4, -0.1, 0.2, 0.0])
    cex_b = np.zeros(6); cex_b[2] = 0.8; cex_b[3] = 0.4   # brows, half-blink
    cap = SyntheticCapture(model, size=320, seed=5)

    donor_yaws = list(np.linspace(-14, 22, n_frames))
    cexs = []
    for i, y in enumerate(donor_yaws):
        e = np.zeros(6)
        e[1] = 0.2 + 0.55 * i / max(n_frames - 1, 1)      # smile grows across burst
        cexs.append(e)

    base = cap.shoot(cid, cex_b, 24.0, person_id="base")
    donors = [cap.shoot(cid, e, y, person_id=f"d{i}",
                        exposure=1.0 + rng.uniform(-0.08, 0.08))
              for i, (y, e) in enumerate(zip(donor_yaws, cexs))]
    swap = LevelBSwap(model, LevelBConfig(mode="pinhole"))
    base_obs = FaceObservation("p0", base.obs_landmarks)

    frames = []
    for i, d in enumerate(donors):
        res = swap.run(base.frame, base_obs,
                       [DonorInput(d.frame, FaceObservation("p0", d.obs_landmarks))])
        frames.append({
            "original": _to_data_uri(d.frame.rgb),
            "result": _to_uri_or_base(res, base),
            "level": "rejected" if res.method == "rejected" else
                     ("A" if res.method.startswith("A") else res.method),
            "coverage": round(res.coverage, 2),
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    html = HTML.replace("__DATA__", __import__("json").dumps(
        [{"name": "Person 1", "frames": frames}]))
    (out_dir / "index.html").write_text(html)
    return out_dir / "index.html"


def _to_uri_or_base(res, base) -> str:
    img = base.rgb if res.method == "rejected" else res.image
    return _to_data_uri(img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="product/picker_out")
    args = ap.parse_args()
    model = SyntheticHeadModel()
    path = build(model, Path(args.out))
    print(f"picker written: {path}")


if __name__ == "__main__":
    main()
