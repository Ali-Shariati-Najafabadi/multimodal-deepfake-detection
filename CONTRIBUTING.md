# Contributing

Thanks for your interest! Contributions are welcome.

- **Bugs and questions:** open an issue and include your config, the command you ran and the full traceback.
- **Pull requests:** keep them focused (one topic per PR) and describe what changed and why. If you change a model or training default, say how it affects results.
- **Code style:** follow the existing style. `python -m compileall .` must pass (it runs in CI).
- **Datasets and weights:** never commit datasets, model weights or personal data. The datasets have their own licenses.

## Ideas that fit the thesis outlook

- Reliability-weighted (e.g. gated) fusion that can down-weight a failing video branch
- Joint training across several datasets
- Segment-level temporal localization
- TimeSformer trained on the full AV-Deepfake1M++ training set
