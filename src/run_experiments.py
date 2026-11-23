"""Run the full experimental pipeline and store results in data/results/."""
import json
import time

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.inspection import permutation_importance

import forecasting as F

OUT = F.DATA / "results"
OUT.mkdir(exist_ok=True)
EVAL_FROM = 2019          # evaluation origins 2019..2024 (targets 2020..2025)
CAL_ORIGINS = range(2013, 2025)
FOCUS = ["Astana IT University", "Al-Farabi Kazakh National University", "L. N. Gumilyov Eurasian National University",
         "Satbayev University", "Nazarbayev University"]

t0 = time.time()
panel = F.build_panel()
fp = F.add_features(panel)
fp.to_csv(OUT / "panel_features.csv", index=False)
print("panel", panel.inst_id.nunique(), "institutions,", len(panel), "rows", panel.groupby("country").inst_id.nunique().to_dict())

if not (OUT / "arima_cache.csv").exists():
    F.compute_arima_cache(fp, range(2012, F.LAST_YEAR + 1)).to_csv(OUT / "arima_cache.csv", index=False)
# always read back from CSV: the text round-trip changes the last float digits, and LightGBM
# splits in the hybrids are sensitive to them, so a cold run must see the same values as a warm one
cache = pd.read_csv(OUT / "arima_cache.csv")
print(f"arima cache {len(cache)} rows, {time.time()-t0:.0f}s")

fc = F.run_backtest(fp, CAL_ORIGINS, cache)
fc = F.add_scale(fc, fp)

# select the hybrid design on the validation origins 2015-2018 (never on the test period)
val = F.summarize(F.score(fc[(fc.origin >= 2015) & (fc.origin < EVAL_FROM)]))
val.to_csv(OUT / "metrics_validation.csv", index=False)
val_mase = val[val.model.isin(["hybrid_res", "hybrid_feat", "hybrid_ens"])].groupby("model").MASE.mean()
chosen = val_mase.idxmin()
print("validation MASE", val_mase.round(3).to_dict(), "-> proposed hybrid =", chosen)
fc = pd.concat([fc, fc[fc.model == chosen].assign(model="hybrid")], ignore_index=True)
F.MODELS = F.MODELS + ("hybrid",)
fc.to_csv(OUT / "backtest_forecasts.csv", index=False)
scored = F.score(fc)
ev = scored[scored.origin >= EVAL_FROM]
ev.to_csv(OUT / "backtest_scored.csv", index=False)

overall = F.summarize(ev)
by_h = F.summarize(ev, by=("indicator", "model", "h"))
kz = F.summarize(ev.merge(panel[["inst_id", "country"]].drop_duplicates(), on="inst_id").query("country == 'KZ'"))
overall.to_csv(OUT / "metrics_overall.csv", index=False)
by_h.to_csv(OUT / "metrics_by_h.csv", index=False)
kz.to_csv(OUT / "metrics_kz.csv", index=False)
print(overall.round(3).to_string())

# young institutions (<= 8 active years at origin) - cold-start regime
young = ev.merge(fp[["inst_id", "year", "age"]].rename(columns={"year": "origin"}), on=["inst_id", "origin"])
F.summarize(young[young.age <= 7]).to_csv(OUT / "metrics_young.csv", index=False)

# paired significance test: hybrid vs every other model on scaled absolute errors
tests = []
for ind in F.INDICATORS:
    base = ev[(ev.indicator == ind) & (ev.model == "hybrid")].set_index(["inst_id", "origin", "h"]).sae
    for m in F.MODELS:
        if m in ("hybrid", chosen):
            continue
        other = ev[(ev.indicator == ind) & (ev.model == m)].set_index(["inst_id", "origin", "h"]).sae
        a, b = base.align(other, join="inner")
        stat, p = wilcoxon(a, b)
        tests.append({"indicator": ind, "vs": m, "median_diff": float(np.median(a - b)),
                      "hybrid_better_share": float((a < b).mean()), "p_value": float(p), "n": len(a)})
pd.DataFrame(tests).to_csv(OUT / "wilcoxon.csv", index=False)

# conformal intervals: plain vs volatility-scaled, evaluated on origins >= EVAL_FROM
cov = []
for ind in F.INDICATORS:
    for scaled in (False, True):
        ci = F.conformal_intervals(fc[fc.indicator == ind], fp, "hybrid", alpha=0.2, scaled=scaled)
        ci = ci[(ci.origin >= EVAL_FROM) & ci.z_lo.notna()]
        inside = (ci.z_target >= ci.z_lo) & (ci.z_target <= ci.z_hi)
        width = F.from_z(ind, ci.z_hi.values) - F.from_z(ind, ci.z_lo.values)
        rel = width / F.from_z(ind, ci.z_target.values)
        for h in F.HORIZONS:
            mk = (ci.h == h).values
            cov.append({"indicator": ind, "method": "scaled" if scaled else "plain", "h": h,
                        "coverage": 100 * inside[mk].mean(), "median_rel_width": 100 * np.median(rel[mk]),
                        "n": int(mk.sum())})
pd.DataFrame(cov).to_csv(OUT / "conformal.csv", index=False)

# feature importance: permutation importance of the h=1 global GBM, averaged over test folds 2019..2024
imp_rows = []
for ind in F.INDICATORS:
    pairs = F.make_pairs(fp, ind, 1, cache)
    per_fold = []
    for t in range(EVAL_FROM, F.LAST_YEAR):
        _, models = F.forecast_origin(pairs, t, 1, return_models=True)
        test = pairs[(pairs.origin == t) & pairs.z_target.notna()]
        r = permutation_importance(models["gbm"], test[F.FEATURES], test.z_target - test.z_origin,
                                   n_repeats=20, random_state=0, scoring="neg_mean_absolute_error")
        per_fold.append(r.importances_mean)
    per_fold = np.array(per_fold)
    for k, f in enumerate(F.FEATURES):
        imp_rows.append({"indicator": ind, "feature": f, "label": F.FEATURE_LABELS[f],
                         "importance": per_fold[:, k].mean(), "std": per_fold[:, k].std()})
pd.DataFrame(imp_rows).to_csv(OUT / "importance.csv", index=False)

# final forecasts from origin 2025 for all institutions, with scaled conformal intervals
final = []
for ind in F.INDICATORS:
    ci_hist = F.conformal_intervals(fc[fc.indicator == ind], fp, "hybrid", alpha=0.2, scaled=True)
    for h in F.HORIZONS:
        pairs = F.make_pairs(fp, ind, h, cache)
        f = F.forecast_origin(pairs, F.LAST_YEAR, h)
        f["hybrid"] = f[chosen]
        res = ci_hist[(ci_hist.h == h) & ci_hist.r.notna()].r.values
        vol = fp[fp.year == F.LAST_YEAR].set_index("inst_id").g_std5_works
        for _, row in f.iterrows():
            v = vol.get(row.inst_id, np.nan)
            v = 0.05 if not np.isfinite(v) else v
            lo, hi = F.interval_from_residuals(row.hybrid, v, res)
            final.append({"indicator": ind, "inst_id": row.inst_id, "year": F.LAST_YEAR + h, "h": h,
                          **{m: float(F.from_z(ind, row[m])) for m in F.MODELS},
                          "lo80": float(F.from_z(ind, lo)), "hi80": float(F.from_z(ind, hi))})
final = pd.DataFrame(final).merge(panel[["inst_id", "name", "country"]].drop_duplicates(), on="inst_id")
final.to_csv(OUT / "forecasts_2026_2028.csv", index=False)
print(final[final.name.isin(FOCUS) & (final.indicator == "works")][["name", "year", "arima", "gbm", "hybrid", "lo80", "hi80"]].round(0).to_string())

summary = {"n_institutions": int(panel.inst_id.nunique()), "n_rows": int(len(panel)),
           "countries": panel.groupby("country").inst_id.nunique().to_dict(),
           "eval_points": int(len(ev) / len(F.MODELS)), "runtime_s": round(time.time() - t0),
           "hybrid_choice": chosen, "validation_mase": val_mase.round(4).to_dict()}
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print(summary)
