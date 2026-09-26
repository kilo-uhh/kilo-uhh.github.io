"""Shared pieces for the Hilo Bay wqb_04 forecast: screening, 3-hourly binning,
climatology lookup, the forecast models, and CRPS. Used by calibrate.py and daily.py
so the live forecast is built exactly the way the backtest was."""
import numpy as np
import pandas as pd
from scipy import stats

STEP = pd.Timedelta("3h")
N_LEAD = 56                      # 56 x 3 h = 7 days
TEMP_COL = "temperature"
QC_COL = "temperature_qc_agg"    # QARTOD: 1 pass, 2 not evaluated, 3 suspect, 4 fail, 9 missing
DROP_QC = [4, 9]
T_RANGE = (18.0, 31.0)
MIN_OBS_PER_BIN = 2
SLOW_WINDOW_BINS = 30 * 8        # 30 days of 3-hourly bins
LEADS = np.arange(1, N_LEAD + 1)


def read_erddap_csv(path_or_url, dedupe=True):
    """dedupe=False for gridded (griddap) data, where many rows share one time."""
    df = pd.read_csv(path_or_url, low_memory=False)
    if str(df.iloc[0]["time"]).strip().upper() == "UTC":   # ERDDAP units row
        df = df.iloc[1:]
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.sort_values("time", kind="stable")
    return (df.drop_duplicates("time") if dedupe else df).set_index("time")


def screen(df, spike_mad=None):
    """Drop QC fails, out-of-range values, flatlines and spikes. Returns (series, spike_mad).
    Pass spike_mad from calibration so short live windows use the same threshold."""
    T = pd.to_numeric(df[TEMP_COL], errors="coerce").dropna()
    bad = ~T.between(*T_RANGE)
    if QC_COL in df.columns:
        bad |= pd.to_numeric(df.loc[T.index, QC_COL], errors="coerce").isin(DROP_QC)
    bad |= T.diff().eq(0).astype(int).rolling(7).sum().eq(7)
    resid = T - T.rolling(25, center=True, min_periods=5).median()
    if spike_mad is None:
        spike_mad = float(1.4826 * resid.abs().median())
    bad |= resid.abs() > 6 * spike_mad
    return T[~bad], spike_mad


def bin_centered(T):
    """3-hourly means centered on 00, 03 ... 21 UTC (window ±1.5 h). These are the targets."""
    b = T.resample(STEP, origin="epoch", offset="-90min").agg(["mean", "count"])
    b.index = b.index + pd.Timedelta("90min")
    return b["mean"].where(b["count"] >= MIN_OBS_PER_BIN).asfreq(STEP)


def bin_trailing(T):
    """Mean of the 3 h *before* each grid time, bins [t − 3 h, t). Used as starting values."""
    b = T.resample(STEP, origin="epoch", closed="left", label="right").agg(["mean", "count"])
    return b["mean"].where(b["count"] >= MIN_OBS_PER_BIN)


def doy_slot(idx):
    return np.minimum(idx.dayofyear.values, 365) - 1, idx.hour.values // 3


def clim_lookup(clim, idx):
    """clim: DataFrame indexed by (doy, slot) with columns mu, sd."""
    d, s = doy_slot(idx)
    mu = clim["mu"].values.reshape(365, 8)
    sd = clim["sd"].values.reshape(365, 8)
    return mu[d, s], sd[d, s]


def make_forecast(issue, obs3h, start, clim, p):
    """All models for one issue time. obs3h: centered 3-hourly series (history up to issue).
    start: trailing-bin mean ending at issue. Returns (long DataFrame in EFI-style columns,
    info dict with the starting value, its anomaly and the 30-day level)."""
    tgt = pd.DatetimeIndex([issue + STEP * h for h in LEADS])
    c_mu, c_sd = clim_lookup(clim, tgt)
    ic_mu, _ = clim_lookup(clim, pd.DatetimeIndex([issue - STEP, issue]))
    anom_ic = start - ic_mu.mean()

    # Slow level L: mean anomaly of centered bins in the past 30 days that end before issue
    hist = obs3h[(obs3h.index <= issue - STEP) & (obs3h.index > issue - STEP * SLOW_WINDOW_BINS)].dropna()
    if len(hist) >= SLOW_WINDOW_BINS // 3:
        h_mu, _ = clim_lookup(clim, hist.index)
        L = float(np.mean(hist.values - h_mu))
    else:
        L = 0.0

    k = LEADS + 0.5
    phi, phi_f = p["phi"], p["phi_f"]
    models = {
        "anomaly_ar1": (c_mu + phi ** k * anom_ic,
                        p["sig_p"] * np.sqrt((1 - phi ** (2 * k)) / (1 - phi ** 2))),
        "anomaly_slow": (c_mu + L + phi_f ** k * (anom_ic - L), np.array(p["sd_slow"])),
        "persistence": (np.full(N_LEAD, start), np.array(p["sd_pers"])),
        "climatology": (c_mu, c_sd),
    }
    rows = [pd.DataFrame({"reference_datetime": issue, "datetime": tgt, "lead_h": LEADS * 3,
                          "model_id": m, "family": "normal", "mu": mu, "sigma": sd})
            for m, (mu, sd) in models.items()]
    info = {"start": float(start), "anom_start": float(anom_ic), "slow_level": L}
    return pd.concat(rows, ignore_index=True), info


def crps_normal(o, m, s):
    z = (o - m) / s
    return s * (z * (2 * stats.norm.cdf(z) - 1) + 2 * stats.norm.pdf(z) - 1 / np.sqrt(np.pi))