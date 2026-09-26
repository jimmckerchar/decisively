# Layar

[![CI](https://github.com/jimmckerchar/layar/actions/workflows/ci.yml/badge.svg)](https://github.com/jimmckerchar/layar/actions/workflows/ci.yml)

Laya/Jev-style "System 1" decisions for Ruby: state in, typed answer + probabilities out.
No text generation, no parsing. Runs locally via ONNX Runtime on [Laya](https://huggingface.co/convaiinnovations/laya)
itself (the default) or on a zero-shot NLI model.

```ruby
Layar.choice(text, options: %w[billing bug account])                  # => Decision(value: "billing", confidence: 0.97, ...)
Layar.bool(text, statement: "Is this message spam, phishing or a scam?") # => Decision(value: true, ...)
Layar.score(text, criterion: "Does the writer express anger?")         # => Decision(value: 0.83, ...)
```

## Installation

```sh
bundle add layar   # or: gem install layar
```

That's all. The first decision downloads the multilingual Laya model (1.3 GB, once) from
[distinctinteractive/laya-onnx](https://huggingface.co/distinctinteractive/laya-onnx) into
`~/.cache/informers`; call `Layar.warm!` at boot to do that up front. It needs ~2 GB of RAM while
loaded. Needs Ruby 3.3+; the `onnxruntime` and `tokenizers` gems it depends on ship prebuilt
binaries, so there is nothing to compile. CI installs the gem from scratch and decides with it
on Linux, macOS (arm64) and Windows.

To download somewhere else, or to run without network access, set `Informers.cache_dir`, or copy
a checkpoint folder from that repository and point `c.laya_model` at it.

## Rails

```ruby
# Gemfile
gem "layar"

# config/initializers/layar.rb
Layar.configure { |c| c.max_options = 20 }
Layar.warm! unless Rails.env.test?

# app/models/ticket.rb
class Ticket < ApplicationRecord
  include Layar::Decidable
  decides :category, from: [:subject, :body], choices: %w[billing bug feature_request account]
end
```

Results are cached in `Rails.cache` automatically.

## Calibration

Raw zero-shot probabilities are overconfident. Fit a temperature on ~100+ labeled examples:

```ruby
Layar.calibrate!(examples)  # => { temperature: 1.85, ece_before: 0.21, ece_after: 0.08 }
```

Persist the temperature and set `c.temperature = 1.85` in the initializer.

## Backends

| | `:laya` (default) | `:nli` |
|---|---|---|
| Model | a [Laya](https://huggingface.co/convaiinnovations/laya) decision model, exported to ONNX | a zero-shot NLI model via `informers`; default `Xenova/bart-large-mnli` |
| Cost of a `choice` | one pass per question, however many options | one pass per option (batched into one call) |
| Download | 1.3 GB (`multilingual`) or 1.7 GB (`english`), on first use | 1.6 GB, on first use |

```ruby
Layar.configure do |c|
  c.laya_model = "english"          # or "multilingual" (default), "owner/repo/subfolder", or a local folder
  # c.backend  = :nli               # to use the NLI model instead
end

Layar.choice(text, options: %w[billing bug account], question: "Which team should handle this?")
Layar.bool(text, statement: "Is this message spam, phishing or a scam?")
```

With `:laya`:
- `choice` asks `question:` (default `c.question`, "What is this about?"). Hash options
  `{ value => description }` are shown to Laya as "value: description".
- `bool` asks each statement as a yes/no question (a question or a claim both work), all in one
  batched call, and is true if any holds.
- `score` is the probability that the criterion holds.

In spot checks on a 4-core CPU (`spec/laya_real_model_spec.rb`, `spec/real_model_spec.rb`), the
multilingual checkpoint answered a 4-option `choice` in ~130 ms against ~540 ms for bart-large-mnli,
barely slowed down with 8 options, and told all 9 spam and ordinary messages apart where
bart-large-mnli managed at best 7. That is a small test set: check accuracy on your own data.

Laya ships overconfident (the multilingual checkpoint has no fitted temperatures), so run
`Layar.calibrate!` on labelled examples before gating on confidence. Broad statements work as
questions or claims ("This message is spam." told all 9 spam and ordinary messages apart), but
narrow ones mislead it as they do NLI ("The message asks you to click a link." flags ordinary requests).

### Yes/no questions: wording, calibration and descriptions

Laya can rank well but score low: it put "This is the third time I've asked. Fix it now." at 0.10
for anger. Three things help, in this order:

1. **Wording.** Try several phrasings on labelled data and keep the best. For anger, "Does the writer
   express anger, irritation or hostility?" beat "Is the writer angry or annoyed?" on the English
   checkpoint, and naming what it is *not* ("…rather than sad or worried?") helped multilingual.
2. **Calibration.** Fit a cut-off on labelled examples; unlike temperature it can move the cut-off,
   in either direction:

   ```ruby
   angry = { statement: "Does the writer express anger, irritation or hostility?" }
   Layar.calibrate_bool!(examples, **angry)   # examples: [{ input: "...", answer: true }, ...]
   # => e.g. { scale: 1.1, shift: 0.9, fitted_for: "3b1f09c2e7d4a8f6", threshold: 0.31, accuracy_before: 0.75, ... }
   Layar.bool(message, **angry)               # applies the stored fit
   ```

   Use 100+ examples: fitted on 20, test accuracy fell to 53-57% on unlucky draws; fitted on
   100, it stayed within a point of fitting on all 200.
3. **`yes:`/`no:` descriptions** (`:laya` only) describe each answer to the model. Test them on your
   data: they helped on hand-written messages but *hurt* ranking on the real comments below.

`threshold` is the raw probability that now maps to 0.5. The fit is stored per statement in
`c.bool_calibrations`; persist it in your initializer:

```ruby
c.bool_calibrations["Does the writer express anger, irritation or hostility?"] =
  { scale: 1.1, shift: 0.9, fitted_for: "3b1f09c2e7d4a8f6" }
```

`fitted_for` fingerprints the model and the `yes:`/`no:` descriptions the fit was made with. If
either changes, `bool` warns once and ignores the fit rather than applying a stale one, so refit.

#### How well it works on real text

On 450 Reddit comments from GoEmotions, labelled by people (`script/eval/anger.rb`; wording chosen
and calibration fitted on 200 separate comments):

| | ranking (AUC) | accuracy | angry caught | calm flagged | sad/worried flagged |
|---|---|---|---|---|---|
| english, best wording, calibrated | 0.85 | 76.9% | 74.5% | 18.0% | 26.0% |
| multilingual, best wording, calibrated | 0.79 | 73.6% | 73.5% | 22.0% | 33.0% |
| multilingual, plain question, raw | 0.78 | 64.7% | 32.0% | 4.0% | 17.0% |
| bart-large-mnli (NLI), calibrated | 0.75 | 67.6% | 69.5% | 26.0% | 46.0% |

Treat anger as a signal to rank or triage by, not a verdict: about a quarter of sad or worried
messages still read as angry, and sarcasm is hard for every model tried. GoEmotions labels are
noisy (people often disagree on "annoyance"), so these figures understate a little, and Reddit is
not your inbox: evaluate on your own messages.

### Exporting Laya to ONNX yourself

Laya publishes PyTorch weights only; Layar downloads ONNX exports of them from
[distinctinteractive/laya-onnx](https://huggingface.co/distinctinteractive/laya-onnx). To build them
yourself (to verify them, or to export a checkpoint you fine-tuned), you need Python 3.10+, about
1 GB of packages and up to ~4 GB of RAM while exporting:

```sh
python -m venv .laya && . .laya/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r script/laya/requirements.txt
python script/laya/export.py multilingual models/laya/multilingual
python script/laya/parity.py multilingual models/laya/multilingual   # ONNX Runtime vs PyTorch
```

| Checkpoint | Size | Notes |
|---|---|---|
| `multilingual` | 1.3 GB | recommended: fastest, 100+ languages, inputs up to 1,024 tokens |
| `english` | 1.7 GB | ModernBERT-large; slower on CPU |
| `typed-decisions` | ~1.7 GB | fine-tuned to Laya's own benchmark; weaker in general use (export untested, not hosted) |

Laya is by Convai Innovations, licensed Apache-2.0. `Layar::Laya` ports its input formatting from
the `laya` 0.3.20 Python package.

## How it differs from Laya

- With `:nli`, Layar rephrases decisions as entailment questions for an off-the-shelf NLI model.
  It matches text on topic more than intent: "Please reset my password" looks like phishing to it.
  Fine-tune an NLI model on your labels and set `c.model`, or use `:laya`.
- With `:laya`, Layar runs Laya's model and input format, but not its Python extras (router,
  hooks, act/escalate head, per-language temperatures).
- Like Laya, keep option sets small; use coarse-to-fine hierarchies for many labels.

## Development

Requires Ruby >= 3.3.

```sh
bundle install
bundle exec rspec        # or: bundle exec rake
```

Specs stub the model, so they run offline without downloading anything. End-to-end specs
against the real model are opt-in (slow; needs ~3 GB RAM for the default model):

```sh
LAYAR_REAL_MODEL=1 bundle exec rspec spec/real_model_spec.rb
LAYAR_REAL_MODEL=1 LAYAR_LAYA_MODELS=models/laya bundle exec rspec spec/laya_real_model_spec.rb
```

The Laya specs replay `spec/fixtures/laya/*.json`, recorded from Python Laya by
`script/laya/record_fixture.py`, so the Ruby port must reproduce its token ids and probabilities.
Run the two files separately on small machines; each model holds ~2 GB while loaded.

`Layar::Decidable` can be used without Rails: `require "layar/decidable"` (needs `activesupport`).
