"""Leave-one-year-out backtest of the live models, saved as the benchmark shown on the page.
Run locally once (and again after each refit):

    python forecast/backtest.py path/to/wqb_04_full.csv

Same method as forecast_v0.py: forecasts issued daily at 00 UTC from a starting window that
ends at issue time; climatology, φ and spreads for year Y come only from the other years.
Writes forecast/benchmark.json: mean CRPS and skill vs climatology (with a 95% interval from
resampling whole years) for every lead.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import core

HERE = Path(__file__).parent
DOY_HALF_WINDOW = 15
N_BOOT = 500
MODELS = ["anomaly_ar1", "anomaly_slow", "persistence", "climatology"]


def main(csv):
    df = core.read_erddap_csv(csv)
    T, _ = core.screen(df)
    y = core.bin_centered(T)
    tt = y.index
    yv = y.values
    icv = core.bin_trailing(T).reindex(tt).values
    year = tt.year.values
    doy, slot = core.doy_slot(tt)
    years = np.unique(year[~np.isnan(yv)])
    leads = core.LEADS
    w = DOY_HALF_WINDOW

    # Climatology, leave-one-year-out
    S = np.zeros((len(years), 365, 8)); S2 = np.zeros_like(S); N = np.zeros_like(S)
    for k, Y in enumerate(years):
        m = (year == Y) & ~np.isnan(yv)
        np.add.at(S[k], (doy[m], slot[m]), yv[m]); np.add.at(S2[k], (doy[m], slot[m]), yv[m] ** 2)
        np.add.at(N[k], (doy[m], slot[m]), 1)

    def smooth(a):
        pad = np.concatenate([a[-w:], a, a[:w]])
        c = np.cumsum(np.vstack([np.zeros((1, 8)), pad]), axis=0)
        return c[2 * w + 1:] - c[:-(2 * w + 1)]

    clim_mu = np.full(len(yv), np.nan); clim_sd = np.full(len(yv), np.nan)
    for k, Y in enumerate(years):
        s, s2, n = (smooth(A.sum(0) - A[k]) for A in (S, S2, N))
        with np.errstate(invalid="ignore", divide="ignore"):
            mu = s / n; sd = np.sqrt(np.maximum(s2 / n - mu ** 2, 0) * n / (n - 1))
        m = year == Y
        clim_mu[m] = mu[doy[m], slot[m]]; clim_sd[m] = sd[doy[m], slot[m]]

    anom = yv - clim_mu
    anom_ic = icv - np.concatenate([[np.nan], 0.5 * (clim_mu[:-1] + clim_mu[1:])])
    L = (pd.Series(anom).shift(1).rolling(core.SLOW_WINDOW_BINS, min_periods=core.SLOW_WINDOW_BINS // 3)
         .mean().fillna(0).values)

    def pairs(x, lag, mask, x0=None):
        x0 = x if x0 is None else x0
        a, b = x0[:-lag], x[lag:]
        ok = ~np.isnan(a) & ~np.isnan(b) & mask[:-lag]
        return a[ok], b[ok]

    params = {}
    for Y in years:
        tr = year != Y
        a0, a1 = pairs(anom, 1, tr); phi = np.sum(a0 * a1) / np.sum(a0 ** 2)
        sig_p = np.std(a1 - phi * a0, ddof=1)
        d0, d1 = pairs(anom - L, 1, tr); phi_f = np.sum(d0 * d1) / np.sum(d0 ** 2)
        sd_pers = np.array([np.std(np.subtract(*pairs(yv, h, tr, icv)[::-1]), ddof=1) for h in leads])
        sd_slow = np.array([np.nanstd((anom[h:] - (L[:-h] + phi_f ** (h + .5) * (anom_ic[:-h] - L[:-h])))
                                      [tr[:-h]], ddof=1) for h in leads])
        params[Y] = dict(phi=phi, sig_p=sig_p, phi_f=phi_f, sd_pers=sd_pers, sd_slow=sd_slow)

    issue_idx = np.where((tt.hour == 0) & ~np.isnan(anom_ic))[0]
    issue_idx = issue_idx[issue_idx + core.N_LEAD < len(yv)]
    k = leads + 0.5
    crps = {m: np.full((len(issue_idx), core.N_LEAD), np.nan) for m in MODELS}
    for r, i in enumerate(issue_idx):
        p = params[year[i]]; tgt = i + leads; obs = yv[tgt]; c = clim_mu[tgt]
        fk = p["phi"] ** k
        mu = {"anomaly_ar1": c + fk * anom_ic[i],
              "anomaly_slow": c + L[i] + p["phi_f"] ** k * (anom_ic[i] - L[i]),
              "persistence": np.full(core.N_LEAD, icv[i]), "climatology": c}
        sd = {"anomaly_ar1": p["sig_p"] * np.sqrt((1 - fk ** 2) / (1 - p["phi"] ** 2)),
              "anomaly_slow": p["sd_slow"], "persistence": p["sd_pers"], "climatology": clim_sd[tgt]}
        for m in MODELS:
            crps[m][r] = core.crps_normal(obs, mu[m], sd[m])

    # Per-year sums so whole years can be resampled
    iy = year[issue_idx]
    yrs = np.unique(iy)
    Ssum = {m: np.array([np.nansum(crps[m][iy == Y], 0) for Y in yrs]) for m in MODELS}
    Ncnt = {m: np.array([np.sum(~np.isnan(crps[m][iy == Y]), 0) for Y in yrs]) for m in MODELS}
    rng = np.random.default_rng(1)
    boots = [rng.integers(0, len(yrs), len(yrs)) for _ in range(N_BOOT)]
    mean = lambda m, idx: Ssum[m][idx].sum(0) / Ncnt[m][idx].sum(0)
    full = np.arange(len(yrs))
    out = {"leads_h": (leads * 3).tolist(), "n_forecasts": int(len(issue_idx)),
           "years": f"{yrs.min()}–{yrs.max()}", "generated": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
           "models": {}}
    for m in MODELS:
        entry = {"crps": np.round(mean(m, full), 4).tolist()}
        if m != "climatology":
            sk = 1 - mean(m, full) / mean("climatology", full)
            bs = np.array([1 - mean(m, b) / mean("climatology", b) for b in boots])
            lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
            entry.update(skill=np.round(sk, 4).tolist(), lo=np.round(lo, 4).tolist(), hi=np.round(hi, 4).tolist())
        out["models"][m] = entry
    (HERE / "benchmark.json").write_text(json.dumps(out))
    s = out["models"]["anomaly_ar1"]["skill"]
    print(f"{len(issue_idx)} backtest forecasts, {out['years']}; anomaly model skill "
          f"3 h {s[0]:.2f}, 1 d {s[7]:.2f}, 3 d {s[23]:.2f}, 7 d {s[55]:.2f}")


if __name__ == "__main__":
    main(sys.argv[1])
