# decide

Small structured decisions from a local LLM **without generating JSON**: every field is asked as `<field>:` after one
shared, cached prompt, and the model answers with a single token, the letter of an option. The token's probability
gives a confidence, a fitted temperature calibrates it, and a threshold routes each decision to
`auto` / `ask` / `escalate`.

Ideas from Codacus / Harsha's "Jev mode" video and bycloud's "Decision Models" video. This repo holds the
experiment code and the measured results. The results are on one PC (RTX 4070 Ti 12 GB) only.

## Files
- `decide.py`: backends (stock `llama-server` with logprobs; Strata, a local engine without logprobs, uses `logit_bias`), the `test`, `calib` and `run` commands.
- `calib.py`: temperature scaling (NLL fit), Brier, ECE, routing, auto threshold, sequence scores for whole-option mode.
- `refit.py`: offline refits (one global temperature vs one per field) from a saved `calibraw_*.json`, no GPU needed.
- `tests/test_calib.py`: 13 acceptance tests (`python -m unittest tests.test_calib`).
- `results/`: the measured reports.

## Use
```
set LLAMA_SERVER=path\to\llama-server.exe
set MODELS_DIR=path\to\gguf\folder        (gemma-4-12b-it-qat-q4_0.gguf)
python decide.py test  --n 20
python decide.py calib --mode letter --n 120 --target 0.95
python decide.py run task.json            # {"system", "schema": {field: [options]}, "states": [...]}
```
```python
from decide import Llama
b = Llama()                                  # loads calib_llama-gemma_letter.json if present
b.decide(system, schema, state)              # {field: (option, calibrated conf, "auto"|"ask"|"escalate")}
```

## Findings (2026-10-15, Gemma 4 12B QAT Q4_0 on llama-server)
**One-token picks vs writing JSON** (20 game-AI states, 4 fields): 429 ms per state, against 974 ms for grammar-constrained
JSON. The two methods agree on 57 of 80 fields. Mean confidence is 0.94 where they agree and 0.82 where they don't.
- The prompt must end in `field:` with **no** trailing space, because the model's token is `" B"`. Thinking must be off.
- Put the shared part first (instructions, fields, options) and the state last, so the KV cache carries across states.
- 8 parallel slots gave no speed-up, because each slot re-reads the prefix.
- With no logprobs (logit_bias only), you get a pick but no confidence, and the picks lean towards option A. On such
  an engine, keep writing JSON.

**Calibration** (`results/calibrep_*`; 120 states labelled by rules, temperature fitted on half and evaluated on the other half, 240 field decisions):

| | letter mode | whole-option text mode |
|---|---|---|
| time per state | 0.56 s | 3.85 s |
| accuracy | 0.892 | 0.542 |
| ECE, no temperature | 0.083 | 0.419 |
| ECE, one global T | 0.091 (T 2.12, worse) | 0.213 (T 7.43) |
| ECE, one T per field | **0.067** | 0.153 |

- One global temperature did not help letter mode, because the fields need opposite corrections. The fitted per-field values are
  action 2.89, stance 2.47, target 0.89 and call_for_help 0.69. `decide()` uses `T_field` when the calibration file has it.
- **Calibration cannot fix confidently wrong answers.** `action` is right only 72% of the time while the model reports about 0.95
  confidence. Even calibrated, the auto tier covers 95% of decisions at an accuracy of 0.91, against a 0.95 target. That needs
  better prompt rules, not a temperature.
- Whole-option text mode (teacher-forced scoring of each option's tokens) is currently **broken**. Its accuracy is the
  same with mean or sum length normalisation, so there is a bug in the scoring code. Don't use it yet.

## Licence
MIT.
