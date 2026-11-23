# What drives the growth of university research output

Code for the paper *Факторы роста публикационной активности университетов Центральной Азии, Южного Кавказа и Монголии: анализ с помощью интерпретируемого машинного обучения* (M. Turbulyak, G. Soltan, Astana IT University), prepared for the Bulletin of K. Zhubanov Aktobe Regional University.

The project asks which properties of a university and its country explain the growth of its research output, using a global LightGBM model, TreeSHAP contributions and a linear panel regression for comparison.

## Data

A panel of 202 universities of Kazakhstan, Uzbekistan, Kyrgyzstan, Tajikistan, Turkmenistan, Azerbaijan, Georgia, Armenia and Mongolia with yearly counts of works and open-access works from OpenAlex (2010–2025) and four World Bank WDI country indicators. The snapshot downloaded on 25 October 2026 is in `data/`. OpenAlex data are CC0, WDI data CC BY 4.0.

## Design

- Target: growth of research output over h = 1 and h = 3 years on the log scale, z(T+h) − z(T).
- Fifteen features known at T: the university's own output history, open-access share and its changes, years active, share of national output, and four country indicators.
- LightGBM with 5-fold cross-validation grouped by university; TreeSHAP contributions (`pred_contrib=True`) are computed only for universities the model did not see.
- Institution-cluster bootstrap intervals for mean |SHAP|; OLS on standardised features with year (and country) effects and errors clustered by university; ablation without country indicators and with the university's own history only, tested with the Wilcoxon signed-rank test.

## Usage

```bash
pip install -r requirements.txt
cd src
python run_experiments.py       # builds data/results/panel_features.csv (and the forecasting backtest)
python growth_drivers.py        # -> data/results/drivers_*.csv
python make_figures_drivers.py  # -> figures/drivers/
```

## Limitations

SHAP contributions describe what the model relies on, not causal effects. Country indicators come from nine countries only, so their contribution cannot be separated reliably from other country differences. The panel keeps universities that were still active in 2025.

## License

MIT, see `LICENSE`.
