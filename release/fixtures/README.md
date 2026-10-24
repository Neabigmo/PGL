# End-to-end regression fixture

`toy01_reference.json` holds the contact tokens of the bundled toy complex, a
fixed synthetic context vector pair and the reference prediction of
`toy_readout/`. `tests/test_regression.py` recomputes the tokens, rebuilds the
1,279-column matrix, loads the readout with `release/schema.json` and compares
the prediction. The readout is a small XGBoost model trained on the 60 toy
complexes; it exists to pin the code-feature-context-readout chain, not to
predict affinity. The context vectors are synthetic and are not encoder output.
