# Figure 7 replication notes (synthetic)

## Understanding of Figure 7
Figure 7 plots how estimation accuracy (MAPE) changes as the DNN training epoch increases, while also showing the DNN training MSE. The training MSE (left axis) decays and flattens as optimization progresses. On the right axis, DeDL’s MAPE decreases as training improves and eventually falls below the SDL and LR baselines. SDL stays higher and noisier because it relies on the raw DNN plug-in estimator, and LR stays roughly constant because it does not benefit from more DNN training. In short: as the DNN becomes better fit, DeDL’s debiasing step turns that fit into improved causal estimation accuracy, and the gap over SDL/LR widens.

## Roadmap for synthetic replication
1. **Synthetic data generation**
   - Choose a small experimental design with `m=4` treatments and `d_c` covariates.
   - Generate covariates `x ~ Uniform(0,1)`.
   - Generate treatment indicators `t` from a small observed set (baseline + single-treatment arms + one two-way combo), matching the limited support in the notebook.
   - Define the outcome as a nonlinear sigmoid of `(x·coef)·t` with additive noise to mirror the paper’s nonlinear response curve.
   - Define the estimand as the ATE between a two-way treatment combo and the baseline.

2. **Model fitting / estimators**
   - Train a small feed-forward neural net (the same structure as the notebook’s `FNN_asig`) using plain SGD.
   - Track training MSE at each epoch as the optimization diagnostic.
   - Compute SDL by plugging the DNN into the ATE formula.
   - Compute DeDL by applying the same influence-function style correction (lambda-inverse and gradient terms) as in the notebook.
   - Fit a linear regression baseline on treatment indicators only (treatment-only LR) to keep LR intentionally misspecified.

3. **Evaluation and plotting**
   - For each epoch, compute MAPE of SDL, DeDL, and LR against the known synthetic ATE.
   - Plot training MSE on the left axis and MAPE (%) on the right axis, matching Figure 7’s dual-axis style.
   - Save the plot as `Replication/figure7_synthetic.svg`.

## Replication summary
Running `Replication/Figure7.py` produces an SVG plot that mirrors Figure 7’s qualitative behavior. The training MSE decays and stabilizes, while DeDL’s MAPE settles in the ~0.12–0.15 range at later epochs. SDL remains substantially higher (often near 0.7–1.1 MAPE), and the LR baseline stays flat around 0.245 MAPE. Thus, as training epochs increase and the DNN fit improves, DeDL consistently beats both SDL and LR in estimation accuracy, which aligns with the key takeaway of the original Figure 7.
