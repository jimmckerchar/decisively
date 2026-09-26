# Changelog

## 0.1.0 (unreleased)

First release.

- `Layar.choice`, `Layar.bool` and `Layar.score`: typed decisions with a probability for every
  answer, running locally via ONNX Runtime. No text generation, nothing to parse.
- Two backends:
  - `:laya` (default): Convai's [Laya](https://huggingface.co/convaiinnovations/laya) decision
    model. One pass per question however many options, several questions batched per call.
    The `multilingual` (default) and `english` checkpoints download on first use from
    [distinctinteractive/laya-onnx](https://huggingface.co/distinctinteractive/laya-onnx), pinned
    to a fixed revision. The Ruby port reproduces the Python package's token ids exactly and its
    probabilities to within 3e-6.
  - `:nli`: any zero-shot NLI model `informers` supports (default `Xenova/bart-large-mnli`), with
    every option scored in one batched call.
- `choice` takes an Array of options or a Hash of `{ value => description }`, and a `question:`
  for Laya. `bool` takes several statements (true if any holds) and, with Laya, `yes:`/`no:`
  descriptions.
- Calibration: `Layar.calibrate!` fits a temperature for `choice`; `Layar.calibrate_bool!` fits a
  per-statement cut-off (Platt scaling) for `bool`, fingerprinted so a stale fit is ignored with a
  warning after the model or descriptions change.
- Rails: `Layar::Decidable` (`decides :category, from:, choices:`) and `Rails.cache` caching.
- `script/laya/`: export Laya checkpoints to ONNX and check them against PyTorch.
  `script/eval/anger.rb`: anger detection measured on GoEmotions.
- Requires Ruby 3.3+.
