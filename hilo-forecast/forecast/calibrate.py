"""Fit the forecast on the full wqb_04 record. Run locally once, then again whenever you
want the model to learn from newer data (e.g. yearly):

    python forecast/calibrate.py path/to/wqb_04_full.csv

Writes forecast/params.json, forecast/climatology.csv, and seeds forecast/archive/obs_3h.csv.
Same method as forecast_v0.py, but fit on all years (no held-out year: this is the live model).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import core

HERE = Path(__file__).parent
DOY_HALF_WINDOW = 15


def main(csv):
    df = core.read_erddap_csv(csv)
    T, spike_mad = core.screen(df)
    y = core.bin_centered(T)
    ic = core.bin_trailing(T).reindex(y.index)
    yv, icv = y.values, ic.values
    d, s = core.doy_slot(y.index)

    # Climatology: day-of-year x 3-hour slot, smoothed over ±15 days (circular)
    ok = ~np.isnan(yv)
    S = np.zeros((365, 8)); S2 = np.zeros((365, 8)); N = np.zeros((365, 8))
    np.add.at(S, (d[ok], s[ok]), yv[ok]); np.add.at(S2, (d[ok], s[ok]), yv[ok] ** 2)
    np.add.at(N, (d[ok], s[ok]), 1)
    w = DOY_HALF_WINDOW

    def smooth(a):
        pad = np.concatenate([a[-w:], a, a[:w]])
        c = np.cumsum(np.vstack([np.zeros((1, 8)), pad]), axis=0)
        return c[2 * w + 1:] - c[:-(2 * w + 1)]

    s_, s2_, n_ = smooth(S), smooth(S2), smooth(N)
    mu = s_ / n_
    sd = np.sqrt(np.maximum(s2_ / n_ - mu ** 2, 0) * n_ / (n_ - 1))
    clim = pd.DataFrame({"doy": np.repeat(np.arange(365), 8), "slot": np.tile(np.arange(8), 365),
                         "mu": mu.ravel(), "sd": sd.ravel()})
    clim.round(4).to_csv(HERE / "climatology.csv", index=False)

    c_mu = mu[d, s]
    anom = yv - c_mu
    anom_ic = icv - np.concatenate([[np.nan], 0.5 * (c_mu[:-1] + c_mu[1:])])
    L = (pd.Series(anom).shift(1)
         .rolling(core.SLOW_WINDOW_BINS, min_periods=core.SLOW_WINDOW_BINS // 3)
         .mean().fillna(0).values)

    def ar_coef(x):
        a, b = x[:-1], x[1:]
        m = ~np.isnan(a) & ~np.isnan(b)
        phi = np.sum(a[m] * b[m]) / np.sum(a[m] ** 2)
        return phi, np.std(b[m] - phi * a[m], ddof=1)

    phi, sig_p = ar_coef(anom)
    phi_f, _ = ar_coef(anom - L)
    sd_pers, sd_slow = [], []
    for h in core.LEADS:
        sd_pers.append(np.nanstd(yv[h:] - icv[:-h], ddof=1))
        pred = L[:-h] + phi_f ** (h + 0.5) * (anom_ic[:-h] - L[:-h])
        sd_slow.append(np.nanstd(anom[h:] - pred, ddof=1))

    params = dict(phi=phi, sig_p=sig_p, phi_f=phi_f, spike_mad=spike_mad,
                  sd_pers=sd_pers, sd_slow=sd_slow,
                  buoy_lat=float(pd.to_numeric(df.get("latitude"), errors="coerce").median())
                  if "latitude" in df else None,
                  buoy_lon=float(pd.to_numeric(df.get("longitude"), errors="coerce").median())
                  if "longitude" in df else None,
                  data_start=str(y.index[0]), data_end=str(y.index[-1]),
                  calibrated_on=pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"))
    params = {k: (round(float(v), 5) if isinstance(v, (float, np.floating)) else
                  [round(float(x), 5) for x in v] if isinstance(v, list) else v)
              for k, v in params.items()}
    (HERE / "params.json").write_text(json.dumps(params, indent=1))

    arch = HERE / "archive"; arch.mkdir(exist_ok=True)
    y.dropna().round(4).rename("observation").rename_axis("datetime").to_csv(arch / "obs_3h.csv")

    print(f"φ = {phi:.3f} (e-folding {-3 / np.log(phi):.1f} h), σ_p = {sig_p:.3f}, "
          f"φ_f = {phi_f:.3f} (e-folding {-3 / np.log(phi_f):.1f} h)")
    print(f"record {y.index[0]} to {y.index[-1]}; {ok.sum()} observed 3-h bins")


if __name__ == "__main__":
    main(sys.argv[1])
