"""Camera ego-motion from Shi-Tomasi corners, pyramidal LK tracking and a RANSAC homography (CPSC 480 L6, L9, L11).

A homography relates two views exactly when the camera only rotates (H = K R K^-1) or the scene is a plane
(L9 p20); walking breaks this, and what is left after H (residual parallax) is the walking proxy."""
import math

import cv2
import numpy as np

from seeforme.perception.egomotion import (EgoSample, box_residual, decision_features, egomotion_series,
                                           estimate_pair, fit_homography, intrinsics, ransac_iterations,
                                           rotation_from_homography, DECISION_FEATURES)

W, H_IMG = 640, 480


def _texture(seed=0, w=W, h=H_IMG):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, (h // 8, w // 8), dtype=np.uint8)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
    return cv2.GaussianBlur(img, (3, 3), 0)


def _rot(yaw_deg, pitch_deg, roll_deg=0.0):
    rv = np.radians([pitch_deg, yaw_deg, roll_deg])        # about camera x (pitch), y (yaw), z (roll)
    return cv2.Rodrigues(rv.reshape(3, 1))[0]


def _angle_deg(Ra, Rb):
    c = (np.trace(Ra @ Rb.T) - 1.0) / 2.0
    return math.degrees(math.acos(min(1.0, max(-1.0, c))))


# (a) pure rotation is recovered through the homography, with 40% outlier matches -------------------------------

def test_rotation_homography_recovered_with_40pct_outliers():
    K = intrinsics(W, H_IMG)
    R = _rot(4.0, -2.0, 0.5)
    Hm = K @ R @ np.linalg.inv(K)
    rng = np.random.default_rng(1)
    n = 300
    p0 = np.column_stack([rng.uniform(0, W, n), rng.uniform(0, H_IMG, n)]).astype(np.float32)
    p1 = cv2.perspectiveTransform(p0.reshape(-1, 1, 2), Hm).reshape(-1, 2)
    p1 += rng.normal(0, 0.3, p1.shape).astype(np.float32)                   # tracking noise
    bad = rng.choice(n, int(0.4 * n), replace=False)
    p1[bad] = np.column_stack([rng.uniform(0, W, len(bad)), rng.uniform(0, H_IMG, len(bad))])
    Hest, inl = fit_homography(p0, p1)
    assert Hest is not None
    assert _angle_deg(rotation_from_homography(Hest, K), R) < 1.0
    assert 0.5 < inl.mean() < 0.7                                           # about the 60% true inliers
    assert inl[bad].mean() < 0.05


def test_rotation_recovered_end_to_end_from_images():
    K = intrinsics(W, H_IMG)
    R = _rot(3.0, 1.5)
    Hm = K @ R @ np.linalg.inv(K)
    a = _texture(2)
    b = cv2.warpPerspective(a, Hm, (W, H_IMG), borderMode=cv2.BORDER_REFLECT)
    s = estimate_pair(a, b, dt=0.5, K=K)
    assert s.reliable
    assert _angle_deg(rotation_from_homography(s.H, K), R) < 1.0
    # yaw +3 deg about +y moves the scene right, i.e. the camera turned left: yaw_rate is negative (deg/s)
    assert abs(s.yaw_rate - (-3.0 / 0.5)) < 1.0
    assert abs(s.pitch_rate - (-1.5 / 0.5)) < 1.0
    assert s.parallax < 1.0 and not s.camera_still


# (b) an independently moving patch has large residual motion; the static background does not ------------------

def test_moving_patch_has_large_residual_background_small():
    bg = _texture(3)
    patch = _texture(4, 120, 120)
    a, b = bg.copy(), bg.copy()
    a[200:320, 260:380] = patch
    b[200:320, 275:395] = patch                                             # patch moves 15 px right
    s = estimate_pair(a, b, dt=0.5, K=intrinsics(W, H_IMG))
    assert s.reliable
    obj = box_residual(s, (275 / W, 200 / H_IMG, 395 / W, 320 / H_IMG))
    back = box_residual(s, (0.0, 0.0, 0.3, 0.35))
    assert obj[1] >= 5 and back[1] >= 5
    assert obj[0] > 8.0                                                     # px at 640 wide
    assert back[0] < 1.0
    assert abs(s.yaw_rate) < 0.5 and abs(s.pitch_rate) < 0.5                # background dominates H
    assert s.camera_still


# (c) degenerate input: unreliable, finite defaults -----------------------------------------------------------

def _finite(s: EgoSample):
    return all(math.isfinite(v) for v in s.as_row())


def test_blank_frame_is_unreliable_and_finite():
    blank = np.full((H_IMG, W), 128, np.uint8)
    for a, b in ((blank, blank), (_texture(5), blank), (blank, _texture(5))):
        s = estimate_pair(a, b, dt=0.5, K=intrinsics(W, H_IMG))
        assert not s.reliable and _finite(s)
        assert s.n_matches < 30 and s.yaw_rate == 0.0 and s.pitch_rate == 0.0


def test_too_few_corners_is_unreliable():
    a = np.full((H_IMG, W), 50, np.uint8)
    for x, y in ((100, 100), (500, 120), (320, 400)):
        cv2.rectangle(a, (x, y), (x + 20, y + 20), 255, -1)                 # 12 corners at most
    b = np.roll(a, 3, axis=1)
    s = estimate_pair(a, b, dt=0.5, K=intrinsics(W, H_IMG))
    assert not s.reliable and _finite(s)
    assert box_residual(s, (0.0, 0.0, 0.1, 0.1)) == (0.0, 0)


def test_decision_features_without_samples_are_zero():
    f = decision_features(np.zeros(0), np.zeros((0, len(EgoSample.FIELDS))), 3.0)
    assert f.shape == (len(DECISION_FEATURES),) and not f.any()


# (d) causality: features at time t use only frames <= t ---------------------------------------------------------

def test_series_is_causal_in_frames():
    K = intrinsics(W, H_IMG)
    base = _texture(6)
    frames = [cv2.warpAffine(base, np.float32([[1, 0, 4 * k], [0, 1, 0]]), (W, H_IMG), borderMode=cv2.BORDER_REFLECT)
              for k in range(6)]
    times = [0.5 * k for k in range(6)]
    t1, f1 = egomotion_series(frames, times, K)
    frames2 = frames[:3] + [_texture(7 + k) for k in range(3)]               # change the future after t = 1.0
    t2, f2 = egomotion_series(frames2, times, K)
    assert np.allclose(t1, times[1:]) and np.allclose(t2, times[1:])
    keep = t1 <= 1.0
    assert np.array_equal(f1[keep], f2[keep])
    assert not np.array_equal(f1[~keep], f2[~keep])


def test_decision_features_use_only_samples_at_or_before_t():
    rng = np.random.default_rng(8)
    t = np.arange(0.5, 10.0, 0.5)
    F = rng.uniform(0, 1, (len(t), len(EgoSample.FIELDS)))
    F[:, EgoSample.FIELDS.index("reliable")] = 1.0
    at = 4.0
    ref = decision_features(t, F, at)
    F2 = F.copy()
    F2[t > at] = rng.uniform(5, 9, F2[t > at].shape)
    assert np.array_equal(ref, decision_features(t, F2, at))
    F3 = F.copy()
    F3[t == at] += 1.0                                                      # the sample at t itself is used
    assert not np.array_equal(ref, decision_features(t, F3, at))


def test_unreliable_samples_are_masked():
    t = np.array([1.0, 1.5, 2.0])
    F = np.ones((3, len(EgoSample.FIELDS)))
    F[:, EgoSample.FIELDS.index("reliable")] = 0.0
    f = dict(zip(DECISION_FEATURES, decision_features(t, F, 2.0)))
    assert f["ego_reliable"] == 0.0 and f["ego_reliable_frac3s"] == 0.0
    for k in ("ego_yaw_abs", "ego_still", "ego_parallax", "ego_yaw_abs_max3s", "ego_parallax_mean3s"):
        assert f[k] == 0.0


# (e) RANSAC iteration count --------------------------------------------------------------------------------------

def test_ransac_iterations_formula():
    assert ransac_iterations(p=0.99, w=0.5, s=4) == 72
    assert ransac_iterations(p=0.99, w=1.0, s=4) == 1
    assert ransac_iterations(p=0.99, w=0.9, s=4) < ransac_iterations(p=0.99, w=0.5, s=4)
