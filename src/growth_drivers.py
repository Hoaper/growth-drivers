"""What drives the growth of university research output? (paper for the Zhubanov bulletin)

Target: growth of research output over h years, z(T+h) - z(T) with z = log(1 + works), h = 1 (main) and 3.
Features: the fifteen features of the forecasting study (forecasting.FEATURES), all known at T.

1. Global LightGBM with the parameters of the forecasting study, evaluated with 5-fold cross-validation grouped by
   institution, so every contribution is computed for universities the model did not see. TreeSHAP contributions
   come from LightGBM itself (predict(..., pred_contrib=True)).
2. Mean |SHAP| per feature with an institution-cluster bootstrap CI, and in two periods (origins up to 2017 and
   from 2018).
3. OLS on standardised features with year effects (and, in a second model, country effects), standard errors
   clustered by institution.
4. Ablation: the same cross-validation without the country indicators and with the institution's own history only;
   Wilcoxon test on institution-mean absolute errors against the full model.

Run after run_experiments.py: python growth_drivers.py   -> data/results/drivers_*.csv
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import wilcoxon
from sklearn.model_selection import GroupKFold

import forecasting as F

RES = F.DATA / "results"
COUNTRY = ["log_gdp_pc", "rd_exp", "tert_enrol", "researchers_pm"]
OWN_HISTORY = ["z_works", "g1_works", "g2_works", "g3_works", "g_mean3_works", "g_std5_works"]
B = 2000
rng = np.random.default_rng(42)

fp = pd.read_csv(RES / "panel_features.csv")


def pairs(h):
    tgt = fp[["inst_id", "year", "z_works"]].copy()
    tgt["year"] -= h
    d = fp.merge(tgt.rename(columns={"z_works": "z_next"}), on=["inst_id", "year"], how="inner")
    d["growth"] = d.z_next - d.z_works
    return d[d.g1_works.notna()].reset_index(drop=True)


def cv_predict(d, features, contrib=False):
    """Out-of-fold predictions (and TreeSHAP contributions) with folds grouped by institution."""
    pred = np.zeros(len(d))
    shap = np.zeros((len(d), len(features) + 1)) if contrib else None
    for tr, te in GroupKFold(n_splits=5).split(d, groups=d.inst_id):
        m = lgb.LGBMRegressor(**F.GBM_PARAMS).fit(d.loc[tr, features], d.loc[tr, "growth"])
        pred[te] = m.predict(d.loc[te, features])
        if contrib:
            shap[te] = m.predict(d.loc[te, features], pred_contrib=True)
    return pred, shap


def boot_mean_abs(values, groups):
    """Institution-cluster bootstrap of the mean of |values| (rows of one institution resampled together)."""
    s = pd.DataFrame(np.abs(values)).groupby(groups.to_numpy()).agg(["sum", "count"])
    sums = s.xs("sum", axis=1, level=1).to_numpy()
    cnt = s.xs("count", axis=1, level=1).to_numpy()[:, 0]
    idx = rng.integers(0, len(cnt), size=(B, len(cnt)))
    means = sums[idx].sum(1) / cnt[idx].sum(1)[:, None]
    return np.quantile(means, 0.025, axis=0), np.quantile(means, 0.975, axis=0)


imp, dep, abl, per, info = [], [], [], [], []
for h in (1, 3):
    d = pairs(h)
    pred, shap = cv_predict(d, F.FEATURES, contrib=True)
    contrib = shap[:, :-1]
    info.append({"h": h, "pairs": len(d), "institutions": d.inst_id.nunique(), "origins": f"{d.year.min()}-{d.year.max()}",
                 "mean_growth": d.growth.mean(), "sd_growth": d.growth.std(),
                 "cv_mae": np.mean(np.abs(pred - d.growth)),
                 "cv_r2": 1 - np.sum((pred - d.growth) ** 2) / np.sum((d.growth - d.growth.mean()) ** 2)})

    lo, hi = boot_mean_abs(contrib, d.inst_id)
    mabs = np.abs(contrib).mean(0)
    for k, f in enumerate(F.FEATURES):
        # direction: correlation between the feature value and its contribution
        ok = d[f].notna().to_numpy()
        rho = pd.Series(d[f][ok]).corr(pd.Series(contrib[ok, k]), method="spearman")
        imp.append({"h": h, "feature": f, "label": F.FEATURE_LABELS[f], "mean_abs_shap": mabs[k],
                    "share": mabs[k] / mabs.sum(), "ci_lo": lo[k], "ci_hi": hi[k], "direction_rho": rho})
        # dependence: SHAP by quantile bin of the feature
        if h == 1:
            q = pd.qcut(d[f], 10, duplicates="drop")
            g = pd.DataFrame({"x": d[f], "s": contrib[:, k], "bin": q}).groupby("bin", observed=True)
            for _, r in g.agg(x=("x", "mean"), s=("s", "mean"), n=("s", "size")).iterrows():
                dep.append({"feature": f, "x": r.x, "shap": r.s, "n": int(r.n)})
        # two periods
        for name, mask in (("2011-2017", d.year <= 2017), ("2018-2024", d.year >= 2018)):
            per.append({"h": h, "feature": f, "period": name, "mean_abs_shap": np.abs(contrib[mask.to_numpy(), k]).mean(),
                        "n": int(mask.sum())})

    # ablation
    err = {"full": np.abs(pred - d.growth)}
    for name, feats in (("no_country", [f for f in F.FEATURES if f not in COUNTRY]), ("own_history", OWN_HISTORY)):
        p, _ = cv_predict(d, feats)
        err[name] = np.abs(p - d.growth)
    err["zero_growth"] = np.abs(d.growth)
    err["mean_growth"] = np.abs(d.growth - d.growth.mean())
    e = pd.DataFrame(err).assign(inst_id=d.inst_id).groupby("inst_id").mean()
    for name in err:
        row = {"h": h, "model": name, "mae": err[name].mean(), "rel_to_full": err[name].mean() / err["full"].mean()}
        if name != "full":
            row.update(p_inst=wilcoxon(e[name], e["full"]).pvalue, full_better_share=(e["full"] < e[name]).mean())
        abl.append(row)

# OLS with standardised features, year effects (+ country effects), errors clustered by institution
d = pairs(1)
cols = F.FEATURES
z = d[cols].apply(lambda s: (s - s.mean()) / s.std())
z = z.fillna(0.0)  # missing values at the mean (the WDI gaps); share reported below
data = pd.concat([z, d[["growth", "year", "country", "inst_id"]]], axis=1)
reg = []
for name, extra in (("year FE", "C(year)"), ("year + country FE", "C(year) + C(country)")):
    m = smf.ols("growth ~ " + " + ".join(cols) + " + " + extra, data).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(data.inst_id)[0]})
    for f in cols:
        reg.append({"model": name, "feature": f, "coef": m.params[f], "se": m.bse[f], "p": m.pvalues[f],
                    "r2": m.rsquared, "n": int(m.nobs)})
missing = d[cols].isna().mean().rename("missing_share").reset_index().rename(columns={"index": "feature"})

pd.DataFrame(info).to_csv(RES / "drivers_info.csv", index=False)
pd.DataFrame(imp).to_csv(RES / "drivers_importance.csv", index=False)
pd.DataFrame(dep).to_csv(RES / "drivers_dependence.csv", index=False)
pd.DataFrame(per).to_csv(RES / "drivers_periods.csv", index=False)
pd.DataFrame(abl).to_csv(RES / "drivers_ablation.csv", index=False)
pd.DataFrame(reg).to_csv(RES / "drivers_regression.csv", index=False)
missing.to_csv(RES / "drivers_missing.csv", index=False)

pd.set_option("display.width", 220)
print(pd.DataFrame(info).round(4).to_string(index=False), "\n")
im = pd.DataFrame(imp)
print(im[im.h == 1].sort_values("mean_abs_shap", ascending=False).round(4).to_string(index=False), "\n")
print(im.pivot_table(index="label", columns="h", values="share").sort_values(1, ascending=False).round(3).to_string(), "\n")
print(pd.DataFrame(per).pivot_table(index=["feature"], columns=["h", "period"], values="mean_abs_shap").round(4).to_string(), "\n")
print(pd.DataFrame(abl).round(4).to_string(index=False), "\n")
print(pd.DataFrame(reg).round(4).to_string(index=False), "\n")
print(missing.round(3).to_string(index=False))
