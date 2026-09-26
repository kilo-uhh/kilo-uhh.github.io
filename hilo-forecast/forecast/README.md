# Hilo Bay water temperature forecast

Daily 7-day forecast of water temperature at PacIOOS buoy wqb_04, on 3-hourly steps
matching the ROMS clock. Same method as the `forecast_v0.py` backtest.

## Files
- `core.py`: screening, binning, models, scoring (shared)
- `calibrate.py`: fits the model on the full record; writes `params.json`, `climatology.csv`,
  and seeds `archive/obs_3h.csv`
- `daily.py`: fetches recent data, issues today's forecast, saves ROMS at the buoy, scores,
  writes `site/latest.json`
- `widget.html`: the page that draws `site/latest.json`
- `archive/`: every forecast issued (`forecasts.csv`), observations (`obs_3h.csv`),
  ROMS forecasts (`roms.csv`). Never edit by hand; this is the scoring record.

## Setup (once)
1. Copy the `forecast/` folder and `.github/workflows/forecast.yml` into the site repo.
2. `pip install -r forecast/requirements.txt`
3. `python forecast/calibrate.py ../data/wqb_04_712b_5843_9069.csv`
4. `python forecast/daily.py` (first live run; check the printed status line)
5. Open `forecast/widget.html` through a local server (`python -m http.server`) to check the page.
6. Commit everything, push, then in GitHub: Actions → "Hilo Bay forecast" → Run workflow.

## Refitting
Rerun `calibrate.py` on a fresh full download (e.g. yearly), and commit. Record the date:
forecasts made before and after a refit come from slightly different models.
