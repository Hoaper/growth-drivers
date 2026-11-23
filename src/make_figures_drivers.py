"""Figures for the growth-drivers paper (Zhubanov, Russian labels): figures/drivers/*.png, 300 dpi."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import forecasting as F

RES = F.DATA / "results"
FIG = F.ROOT / "figures" / "drivers"
FIG.mkdir(parents=True, exist_ok=True)

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
BLUE, ORANGE, GRAY1, GRAY2 = "#2a78d6", "#eb6834", "#8a8984", "#c9c8c1"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 9,
    "axes.edgecolor": GRAY1, "axes.labelcolor": INK, "axes.titlesize": 9.5, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED, "legend.frameon": False,
    "legend.fontsize": 8, "lines.linewidth": 1.6, "savefig.dpi": 300, "savefig.bbox": "tight",
})
RU = {"z_works": "Объём публикаций (лог.)", "g1_works": "Рост публикаций, t", "g2_works": "Рост публикаций, t−1",
      "g3_works": "Рост публикаций, t−2", "g_mean3_works": "Средний рост за 3 года",
      "g_std5_works": "Волатильность роста, 5 лет", "z_oa": "Доля открытого доступа (логит)",
      "g1_oa": "Изменение доли ОД, t", "g2_oa": "Изменение доли ОД, t−1", "age": "Лет активности",
      "log_country_share": "Доля в национальном выпуске", "log_gdp_pc": "ВВП на душу населения (лог.)",
      "rd_exp": "Расходы на НИОКР, % ВВП", "tert_enrol": "Охват высшим образованием, %",
      "researchers_pm": "Исследователей на 1 млн жителей"}
COUNTRY = {"log_gdp_pc", "rd_exp", "tert_enrol", "researchers_pm"}

imp = pd.read_csv(RES / "drivers_importance.csv")
dep = pd.read_csv(RES / "drivers_dependence.csv")

# 1. mean |SHAP| with bootstrap CI, h = 1
d = imp[imp.h == 1].sort_values("mean_abs_shap")
fig, ax = plt.subplots(figsize=(5.6, 3.8))
cols = [ORANGE if f in COUNTRY else BLUE for f in d.feature]
ax.barh([RU[f] for f in d.feature], d.mean_abs_shap, color=cols,
        xerr=[d.mean_abs_shap - d.ci_lo, d.ci_hi - d.mean_abs_shap], error_kw=dict(ecolor=MUTED, lw=0.8))
ax.set_xlabel("Средний |SHAP|, вклад в годовой рост (лог. ед.)")
ax.grid(axis="y", visible=False)
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)],
          labels=["признаки вуза", "показатели страны"], loc="lower right")
fig.tight_layout()
fig.savefig(FIG / "fig1_importance.png")
plt.close(fig)

# 2. dependence: mean SHAP by decile of the feature, four institution features
panels = [("z_works", lambda z: np.expm1(z), "Публикаций в год (лог. шкала)", True),
          ("g1_works", lambda g: 100 * np.expm1(g), "Рост публикаций за последний год, %", False),
          ("z_oa", lambda z: 100 / (1 + np.exp(-z)), "Доля открытого доступа, %", False),
          ("log_country_share", lambda z: 100 * np.exp(z), "Доля в национальном выпуске, % (лог. шкала)", True)]
fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.6))
for ax, (f, tr, xl, logx) in zip(axes.ravel(), panels):
    g = dep[dep.feature == f]
    ax.axhline(0, color=GRAY2, lw=1)
    ax.plot(tr(g.x), g.shap, color=BLUE, marker="o", ms=3)
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xl)
    ax.set_ylabel("Средний SHAP")
    ax.set_title(RU[f])
fig.tight_layout()
fig.savefig(FIG / "fig2_dependence.png")
plt.close(fig)
print("saved", sorted(p.name for p in FIG.glob("*.png")))
