"""Numerical definitions in Chapter 3. Spatial arrays use (y,x); flow components use (x,y)."""
import numpy as np
import cv2
from scipy import signal


GROUPS = {"A": 16, "B": 120, "C": 24, "D": 84, "E": 48, "F": 32}


def feature_names(lags):
    names = ["A_" + k for k in ("ax_mean", "ay_mean", "ax_std", "ay_std", "speed_q25",
             "speed_q50", "speed_q75", "speed_mean", "speed_all", "dx", "dy",
             "direction_consistency", "delta_mean", "delta_std", "aspect_mean", "aspect_std")]
    names += [f"B_cell{k:02d}_dir{45*j:03d}" for k in range(12) for j in range(8)]
    names += [f"B_cell{k:02d}_{stat}" for stat in ("mean", "std") for k in range(12)]
    names += [f"C_row{r}_{s}" for r in range(6) for s in ("balance", "pearson", "in", "out")]
    names += [f"D_row{r}_lag{lag:02d}" for r in range(6) for lag in lags]
    names += [f"D_row{r}_{s}" for r in range(6) for s in ("peak_lag_norm", "peak_r")]
    names += [f"D_row{r}_quarter{q}" for r in range(6) for q in range(4)]
    names += [f"E_{kind}{k}_{s}" for kind in ("band", "cell") for s in ("mean", "std") for k in range(8)]
    names += [f"E_graybin{j:02d}" for j in range(16)]
    names += [f"F_cell{k:02d}_{s}" for s in ("mean", "std") for k in range(16)]
    assert len(names) == len(set(names)) == sum(GROUPS.values())
    return names


class PairData:
    """Indices always refer to the original sequence of N = T - 1 frame pairs."""
    def __init__(self, n):
        self.n = n
        self.good = np.zeros(n, bool)
        self.a = np.zeros((n, 2))
        self.center = np.zeros((n, 2))
        self.height = np.zeros(n)
        self.aspect = np.zeros(n)
        self.hist = np.zeros((12, 8))
        self.energy = np.zeros((n, 12))
        self.b_support = np.zeros((n, 12), bool)
        self.sides = np.zeros((n, 6, 2))
        self.coord = np.zeros((n, 6, 3))
        self.shape = np.zeros((n, 32))
        self.change = np.full((n, 16), np.nan)


def reference_motion(flow, mask, box, cfg):
    """3.2.1–3.2.3. Median by component on the masked central patch and background."""
    x, y, w, h = box
    x0, x1 = x + np.ceil(w * np.array([cfg["x_min"], cfg["x_max"]])).astype(int)
    y0, y1 = y + np.ceil(h * np.array([cfg["y_min"], cfg["y_max"]])).astype(int)
    body = flow[y0:y1, x0:x1][mask[y0:y1, x0:x1] != 0]
    if len(body) < cfg["min_body_pixels"]:
        return None, "insufficient_body_reference"
    halo = cfg["background_halo"]
    bg = np.ones(mask.shape, bool)
    bg[max(0,y-halo):y+h+halo, max(0,x-halo):x+w+halo] = False
    if np.count_nonzero(bg) < cfg["min_background_pixels"]:
        return None, "insufficient_background"
    g = np.median(body, axis=0)
    c = np.median(flow[bg], axis=0)
    height = max(h, cfg["height_floor"])
    center = (x + w * (cfg["x_min"] + cfg["x_max"]) / 2,
              y + h * (cfg["y_min"] + cfg["y_max"]) / 2)
    relative = (flow[y:y+h, x:x+w] - g) / height
    return ((g-c)/height, relative, center, height), ""


def cell_ids(h, w):
    return (np.arange(h)[:, None] * 6 // h) * 2 + np.arange(w)[None, :] * 2 // w


def regional_motion(relative, person, data, t, eps):
    """3.3.2–3.3.3. B uses the unexpanded mask. C uses a clipped 7x7 dilation."""
    cells = cell_ids(*person.shape)
    mag = np.hypot(relative[..., 0], relative[..., 1])
    inside = person != 0
    active = cells[inside]
    weights = mag[inside].astype(float)
    flow = relative[inside]
    bins = np.mod(np.arctan2(flow[:, 1].astype(float), flow[:, 0]), 2*np.pi) * (4/np.pi)
    low = np.floor(bins).astype(int)
    frac = bins - low
    for direction, share in ((low, 1-frac), (low+1, frac)):
        idx = active * 8 + direction % 8
        data.hist += np.bincount(idx, weights=weights*share, minlength=96).reshape(12, 8)
    counts = np.bincount(active, minlength=12)
    data.b_support[t] = counts > 0
    data.energy[t] = np.bincount(active, weights=weights, minlength=12) / np.maximum(counts, 1)

    support = cv2.dilate(person, np.ones((7, 7), np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0) != 0
    area = np.maximum(np.bincount(cells.ravel(), minlength=12), 1)
    def density(values):
        return (np.bincount(cells[support], weights=values[support], minlength=12) / area).reshape(6, 2)
    sides = density(mag)
    pos = density(np.maximum(relative[..., 0], 0))
    neg = density(np.maximum(-relative[..., 0], 0))
    total = np.maximum((pos + neg).sum(axis=1), eps)
    data.sides[t] = sides
    data.coord[t, :, 0] = sides.min(axis=1) / np.maximum(sides.max(axis=1), eps)
    data.coord[t, :, 1] = 2 * np.minimum(pos[:, 0], neg[:, 1]) / total
    data.coord[t, :, 2] = 2 * np.minimum(neg[:, 0], pos[:, 1]) / total


def appearance(gray_person, person):
    """3.3.5. Eight widths, eight masked gray sums / whole cell area, sixteen frequencies."""
    gray = cv2.resize(gray_person, (32, 32), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(person, (32, 32), interpolation=cv2.INTER_NEAREST) != 0
    width = mask.reshape(8, 4, 32).any(axis=1).mean(axis=1)
    layout = ((gray / 255) * mask).reshape(4, 8, 2, 16).mean(axis=(1, 3)).ravel()
    hist = np.bincount(gray[mask] // 16, minlength=16) / max(1, int(mask.sum()))
    return np.r_[width, layout, hist], bool(mask.any())


def changes(gray, following, box, cfg):
    """3.3.6. Round real boundaries only after partition; compare original pixels."""
    result = np.full(16, np.nan)
    if box is None or box[3] < cfg["min_height"]:
        return result
    x, y, w, h = box
    width = max(cfg["min_width_heights"]*h, w + cfg["extra_width_heights"]*h)
    cx = x + w/2
    xe = np.clip(np.ceil(np.linspace(cx-width/2, cx+width/2, 5)).astype(int), 0, gray.shape[1])
    ye = np.clip(np.ceil(np.linspace(y-cfg["top_pad"]*h, y+h+cfg["bottom_pad"]*h, 5)).astype(int), 0, gray.shape[0])
    changed = cv2.absdiff(gray, following) > cfg["threshold"]
    for row in range(4):
        for col in range(4):
            cell = changed[ye[row]:ye[row+1], xe[col]:xe[col+1]]
            if cell.size:
                result[4*row+col] = cell.mean()
    return result


def runs(indices):
    """Return half-open runs of consecutive true entries."""
    edges = np.diff(np.r_[False, indices, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))


def pearson(x, y, eps):
    x = x - x.mean()
    y = y - y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom <= eps:
        return np.nan
    return float(np.clip(np.dot(x, y) / denom, -1, 1))


def translation(data, eps):
    """3.3.1. q and d use N, not |G|. Differencing never bridges invalid frames."""
    g = data.good
    if not g.any():
        return np.full(16, np.nan)
    a = data.a[g]
    speed = np.linalg.norm(a, axis=1)
    adjacent = g[1:] & g[:-1]
    d = np.full(2, np.nan)
    delta_stats = [np.nan, np.nan]
    if adjacent.any():
        d = (np.diff(data.center, axis=0)[adjacent] / data.height[1:][adjacent, None]).sum(axis=0) / data.n
        delta = np.linalg.norm(np.diff(data.a, axis=0)[adjacent], axis=1)
        delta_stats = [delta.mean(), delta.std()]
    return np.r_[a.mean(axis=0), a.std(axis=0), np.quantile(speed, [.25, .5, .75], method="linear"),
                 speed.mean(), speed.sum()/data.n, d, np.linalg.norm(a.sum(axis=0))/max(speed.sum(), eps),
                 delta_stats, data.aspect[g].mean(), data.aspect[g].std()]


def temporal(data, cfg, eps):
    """3.3.4. Original-time quarters, lagged Pearson on the earliest longest run."""
    lags = cfg["lags"]
    corr = np.full((6, len(lags)), np.nan)
    peak = np.full((6, 2), np.nan)
    quarters = np.full((6, 4), np.nan)
    z = data.energy.reshape(data.n, 6, 2).mean(axis=2)
    for q, indices in enumerate(np.array_split(np.arange(data.n), 4)):
        usable = indices[data.good[indices]]
        if usable.size:
            quarters[:, q] = z[usable].mean(axis=0)
    start, stop = max(runs(data.good), key=lambda run: run[1]-run[0], default=(0, 0))
    n = stop - start
    status = ["run_too_short"] * 6
    limit = min(max(lags), n//2)
    if n >= cfg["min_run"]:
        series = signal.detrend(z[start:stop], axis=0, type="linear")
        for row, s in enumerate(series.T):
            if s.std() < cfg["min_detrended_std"]:
                status[row] = "low_detrended_variance"
                continue
            curve = np.full(limit+1, np.nan)
            for lag in range(3, limit+1):
                curve[lag] = pearson(s[:-lag], s[lag:], eps)
            for i, lag in enumerate(lags):
                if lag <= limit:
                    corr[row, i] = curve[lag]
            # On ties keep the shorter lag; never use an unavailable neighbor.
            candidates = []
            for a, b in runs(np.isfinite(curve)):
                peaks = signal.find_peaks(curve[a:b])[0] + a
                candidates.extend(int(k) for k in peaks if curve[k] > 0)
            status[row] = "no_positive_local_peak" if np.isfinite(curve).any() else "undefined_pearson"
            if candidates:
                k = max(candidates, key=curve.__getitem__)
                peak[row] = [k/max(lags), curve[k]]
                status[row] = "ok"
    return np.r_[corr.ravel(), peak.ravel(), quarters.ravel()], (start, stop), status


def summarize(data, cfg):
    eps = cfg["epsilon"]
    g = data.good
    a = translation(data, eps)
    b = np.full(120, np.nan)
    c = np.full((6, 4), np.nan)
    e = np.full(48, np.nan)
    if g.any():
        energy = data.energy[g]
        b = np.r_[data.hist.ravel()/max(data.hist.sum(), eps), energy.mean(axis=0), energy.std(axis=0)]
        c[:, [0, 2, 3]] = data.coord[g].mean(axis=0)
        sides = data.sides[g]
        for row in range(6):
            c[row, 1] = pearson(sides[:, row, 0], sides[:, row, 1], eps)
        shape = data.shape[g]
        e = np.r_[shape[:, :8].mean(axis=0), shape[:, :8].std(axis=0),
                  shape[:, 8:16].mean(axis=0), shape[:, 8:16].std(axis=0), shape[:, 16:].mean(axis=0)]
    d, segment, peak_status = temporal(data, cfg["temporal"], eps)
    f = np.full((2, 16), np.nan)
    for k, column in enumerate(data.change.T):
        observed = column[np.isfinite(column)]
        if observed.size:
            f[:, k] = [observed.mean(), observed.std()]
    raw = np.r_[a, b, c.ravel(), d, e, f.ravel()]
    if raw.shape != (324,) or np.isinf(raw).any():
        raise ArithmeticError("invalid vector size or overflow")
    return np.nan_to_num(raw).astype(np.float32), np.isfinite(raw), segment, peak_status
