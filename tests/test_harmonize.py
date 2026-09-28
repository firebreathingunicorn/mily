import numpy as np

from besttake.level_b.harmonize import (color_transfer, estimate_noise_sigma,
                                        grain_match, shading_transfer)


def test_noise_sigma_estimate():
    rng = np.random.default_rng(0)
    img = rng.normal(0.5, 0.08, (128, 128, 3)).astype(np.float32)
    img = np.clip(img, 0, 1)
    sigma = estimate_noise_sigma(img)
    assert 0.05 < sigma < 0.12, f"sigma {sigma}"


def test_color_transfer_matches_gain_and_bias():
    rng = np.random.default_rng(1)
    base = rng.uniform(0.2, 0.8, (96, 96, 3)).astype(np.float32)
    pasted = np.clip(base * 0.7 + 0.05, 0, 1).astype(np.float32)
    w = np.ones((96, 96), np.float32)
    out, params = color_transfer(pasted, base, w, strength=1.0)
    # overlap stats should now match the base closely
    assert np.abs(out.mean(0).mean(0) - base.mean(0).mean(0)).max() < 0.02
    assert params["n"] > 1000


def test_shading_transfer_recovers_ratio():
    rng = np.random.default_rng(2)
    pasted = rng.uniform(0.3, 0.6, (96, 96, 3)).astype(np.float32)
    base = (pasted * 1.3).astype(np.float32)
    w = np.ones((96, 96), np.float32)
    out, gain = shading_transfer(pasted, base, w, sigma_px=12.0, strength=1.0)
    assert 1.2 < float(np.nanmean(gain)) < 1.4
    assert np.abs(out - base).mean() < 0.05


def test_shading_transfer_leaves_base_alone():
    """Pixels outside the pasted support keep gain 1 (never touched)."""
    pasted = np.full((96, 96, 3), 0.5, np.float32)
    base = np.full((96, 96, 3), 0.9, np.float32)
    w = np.zeros((96, 96), np.float32)
    w[20:70, 20:70] = 1.0
    out, gain = shading_transfer(pasted, base, w, sigma_px=8.0, strength=1.0)
    assert (gain[~(w > 0.02)] == 1.0).all()


def test_grain_match_only_adds_deficit():
    rng = np.random.default_rng(3)
    out = np.full((64, 64, 3), 0.4, np.float32)
    w = np.ones((64, 64), np.float32)
    # donor noisier than base: nothing to add
    same = grain_match(out.copy(), w, 0.01, 0.02, rng)
    assert np.array_equal(same, out)
    # base noisier: grain added inside the weight
    add = grain_match(out.copy(), w, 0.03, 0.01, rng)
    assert add.std() > out.std()
