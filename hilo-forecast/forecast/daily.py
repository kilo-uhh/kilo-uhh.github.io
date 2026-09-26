"""Daily Hilo Bay forecast: fetch recent wqb_04 data, issue today's 00 UTC forecast,
archive it, save ROMS's forecast at the buoy, score past forecasts, write site/latest.json.

    python forecast/daily.py                      # live, from PacIOOS ERDDAP
    python forecast/daily.py --local recent.csv   # test with a local ERDDAP-format file
    python forecast/daily.py --now 2026-09-20T02:15Z --local full.csv   # pretend it's another day

Safe to run more than once a day: an issue time already in the archive is not re-issued.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import core

HERE = Path(__file__).parent
ARCH = HERE / "archive"
SITE = HERE / "site"
ERDDAP = "https://pae-paha.pacioos.hawaii.edu/erddap"
FETCH_DAYS = 45                  # enough for the 30-day slow level and 7 days of scoring
ROMS_DATASET, ROMS_VAR, ROMS_DEPTH = "roms_hiig", "temp", 1.0
ROMS_SEARCH_DEG = 0.15           # search box half-width (~16 km) for the nearest ocean cell
SCORE_WINDOW_DAYS = 90
Z90 = 1.645                      # half-width of a 90% normal range, in sigmas


def fetch_buoy(since, local=None):
    if local:
        return core.read_erddap_csv(local)
    q = f"time,{core.TEMP_COL},{core.QC_COL}&time%3E={since:%Y-%m-%dT%H:%M:%SZ}"
    return core.read_erddap_csv(f"{ERDDAP}/tabledap/wqb_04.csv?{q}")


def erddap_get(url):
    """Fetch an ERDDAP CSV; on failure raise RuntimeError with ERDDAP's own message."""
    import io
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen(url, timeout=120) as resp:
            return core.read_erddap_csv(io.StringIO(resp.read().decode()), dedupe=False)
    except urllib.error.HTTPError as e:
        msg = e.read().decode(errors="replace")
        detail = next((l for l in msg.splitlines() if "rror" in l), msg[:200]).strip()
        raise RuntimeError(f"HTTP {e.code}: {detail[:200]}") from None


def roms_cell(lat, lon):
    """Nearest ROMS ocean cell to the buoy. Searched once, then cached in archive/roms_cell.json."""
    path = ARCH / "roms_cell.json"
    if path.exists():
        return json.loads(path.read_text())
    box = ROMS_SEARCH_DEG
    q = (f"{ROMS_VAR}[(last)][({ROMS_DEPTH})][({lat - box}):1:({lat + box})]"
         f"[({lon - box}):1:({lon + box})]")
    g = erddap_get(f"{ERDDAP}/griddap/{ROMS_DATASET}.csv?{q}")
    g = g.reset_index(drop=True)
    g = pd.DataFrame({"v": pd.to_numeric(g[ROMS_VAR], errors="coerce"),
                      "la": pd.to_numeric(g["latitude"], errors="coerce"),
                      "lo": pd.to_numeric(g["longitude"], errors="coerce")}).dropna()
    if g.empty:
        raise RuntimeError(f"no ocean cells within {box}° of the buoy")
    km = 111.2 * np.hypot(g.la - lat, (g.lo - lon) * np.cos(np.radians(lat)))
    best = g.loc[km.idxmin()]
    cell = {"lat": float(best.la), "lon": float(best.lo), "km_from_buoy": round(float(km.min()), 2),
            "depth_m": ROMS_DEPTH, "ocean_cells_in_box": int(len(g))}
    path.write_text(json.dumps(cell, indent=1))
    return cell


def fetch_roms(issue, cell):
    """ROMS at the chosen cell, from issue time to the end of ROMS's current forecast, trimmed to 7 days."""
    q = (f"{ROMS_VAR}[({issue:%Y-%m-%dT%H:%M:%SZ}):1:(last)]"
         f"[({ROMS_DEPTH})][({cell['lat']})][({cell['lon']})]")
    r = erddap_get(f"{ERDDAP}/griddap/{ROMS_DATASET}.csv?{q}")
    r = r[r.index <= issue + core.STEP * core.N_LEAD]
    return pd.DataFrame({"reference_datetime": issue, "datetime": r.index,
                         "roms_temp": pd.to_numeric(r[ROMS_VAR], errors="coerce").values})


def upsert(path, new, keys):
    if path.exists():
        old = pd.read_csv(path)
        for k in keys:
            if "datetime" in k:
                old[k] = pd.to_datetime(old[k], format="ISO8601", utc=True)
        new = pd.concat([old, new]).drop_duplicates(keys, keep="last")
    new.sort_values(keys).to_csv(path, index=False)
    return new


def main(now=None, local=None):
    now = pd.Timestamp(now or pd.Timestamp.now(tz="UTC")).tz_convert("UTC")
    issue = now.floor("D")                                   # today 00 UTC (14:00 HST)
    p = json.loads((HERE / "params.json").read_text())
    clim = pd.read_csv(HERE / "climatology.csv")
    ARCH.mkdir(exist_ok=True); SITE.mkdir(exist_ok=True)
    status = []

    # 1. New observations: screen, bin, and add finished bins to the archive
    raw = fetch_buoy(issue - pd.Timedelta(days=FETCH_DAYS), local)
    raw = raw[(raw.index >= issue - pd.Timedelta(days=FETCH_DAYS)) & (raw.index < now)]
    T, _ = core.screen(raw, spike_mad=p["spike_mad"])
    y_new = core.bin_centered(T)
    y_new = y_new[y_new.index + pd.Timedelta("90min") <= now].dropna()   # window closed
    obs = upsert(ARCH / "obs_3h.csv",
                 y_new.round(4).rename("observation").rename_axis("datetime").reset_index(),
                 ["datetime"])
    obs3h = obs.set_index("datetime")["observation"].sort_index()
    last_obs = T.index.max() if len(T) else None

    # 2. Today's forecast, if the starting window [21:00, 00:00) UTC has data
    fpath = ARCH / "forecasts.csv"
    done = (pd.to_datetime(pd.read_csv(fpath, usecols=["reference_datetime"])["reference_datetime"],
                           format="ISO8601", utc=True).eq(issue).any() if fpath.exists() else False)
    start = core.bin_trailing(T).get(issue, np.nan)
    L = None
    if done:
        status.append("already issued")
    elif np.isnan(start):
        status.append("no data in the 3 h before issue: forecast skipped")
    else:
        fc, L = core.make_forecast(issue, obs3h, float(start), clim, p)
        upsert(fpath, fc.round({"mu": 4, "sigma": 4}), ["reference_datetime", "model_id", "datetime"])
        status.append(f"issued (start {start:.2f} °C, 30-day level {L:+.2f} °C)")

    # 3. ROMS forecast at the buoy, saved the same day so it can be scored later
    if p.get("buoy_lat") is not None:
        try:
            cell = roms_cell(p["buoy_lat"], p["buoy_lon"])
            r = fetch_roms(issue, cell)
            if r["roms_temp"].notna().any():
                upsert(ARCH / "roms.csv", r.round({"roms_temp": 4}), ["reference_datetime", "datetime"])
                status.append(f"ROMS saved (cell {cell['km_from_buoy']} km from buoy)")
            else:
                status.append("ROMS returned no values at the chosen cell")
        except Exception as e:                               # never let ROMS break the forecast
            status.append(f"ROMS fetch failed: {type(e).__name__}: {e}"[:300])
    else:
        status.append("no buoy location in params.json: ROMS skipped")

    # 4. Score every archived forecast that now has an observation
    allfc = None
    if fpath.exists():
        allfc = pd.read_csv(fpath)
        for k in ["reference_datetime", "datetime"]:
            allfc[k] = pd.to_datetime(allfc[k], format="ISO8601", utc=True)
    scores = []
    if allfc is not None:
        sc = allfc.merge(obs.rename(columns={"observation": "obs"}), on="datetime")
        sc = sc[sc["reference_datetime"] >= issue - pd.Timedelta(days=SCORE_WINDOW_DAYS)]
        sc["crps"] = core.crps_normal(sc["obs"], sc["mu"], sc["sigma"])
        for (m, h), g in sc[sc.lead_h.isin([3, 24, 72, 168])].groupby(["model_id", "lead_h"]):
            scores.append({"model_id": m, "lead_h": int(h), "crps": round(g.crps.mean(), 3),
                           "n": int(len(g))})

    # 5. Page data
    cur = allfc[allfc.reference_datetime == allfc.reference_datetime.max()] if allfc is not None else None
    recent = obs3h[(obs3h.index >= issue - pd.Timedelta(days=7)) & (obs3h.index <= now)]
    iso = lambda t: pd.Timestamp(t).isoformat()
    out = {
        "generated": iso(now), "status": status,
        "last_observation": iso(last_obs) if last_obs is not None else None,
        "issued": iso(cur.reference_datetime.iloc[0]) if cur is not None and len(cur) else None,
        "observations": [[iso(t), round(v, 3)] for t, v in recent.items()],
        "forecast": {m: [[iso(r.datetime), round(r.mu, 3), round(r.mu - Z90 * r.sigma, 3),
                          round(r.mu + Z90 * r.sigma, 3)] for r in g.itertuples()]
                     for m, g in cur.groupby("model_id")} if cur is not None else {},
        "scores": scores, "score_window_days": SCORE_WINDOW_DAYS,
        "model": {k: p[k] for k in ["phi", "phi_f", "data_start", "data_end", "calibrated_on"]},
    }
    rp = ARCH / "roms.csv"
    if rp.exists() and out["issued"]:
        rr = pd.read_csv(rp)
        for k in ["reference_datetime", "datetime"]:
            rr[k] = pd.to_datetime(rr[k], format="ISO8601", utc=True)
        rr = rr[rr.reference_datetime == pd.Timestamp(out["issued"])]
        out["roms"] = [[iso(t), round(v, 3)] for t, v in zip(rr.datetime, rr.roms_temp) if pd.notna(v)]
    (SITE / "latest.json").write_text(json.dumps(out))
    print(f"{iso(now)}: " + "; ".join(status))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--now"); ap.add_argument("--local")
    a = ap.parse_args()
    main(a.now, a.local)