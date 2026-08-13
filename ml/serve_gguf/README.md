# GGUF Serving and Constrained Decoding Pipeline

This directory quantizes the FT-JRN adapter into GGUF Q4_K_M format for CPU/edge inference.

Unlike the GPU-based 4-bit serving path in `services/model_server/` (which relies on
prompting alone to produce valid schema JSON), this path uses `llama.cpp`'s grammar-
constrained decoding. A GBNF grammar forces the model's output to structurally match the
`valence` / `arousal` / `emotion_tags` schema at the token level, rather than hoping the
model follows the prompt.

## Pipeline

### 1. Merge adapter
Merges the PEFT LoRA adapter into the base model's full bf16 weights, on CPU.
```bash
uv run python ml/serve_gguf/merge_adapter.py
```

### 2. Quantize to GGUF
Converts the merged HF model to f16 GGUF (via the vendored llama.cpp converter script),
then quantizes f16 -> Q4_K_M via `llama_cpp`'s low-level Python API.
```bash
uv run python ml/serve_gguf/quantize_gguf.py
```

### 3. Evaluate
Runs each validation example through the quantized GGUF model twice — once unconstrained,
once with `grammar.gbnf` — and reports valid-JSON rate, valence/arousal MAE and correlation,
and average latency side by side. Logs to the same MLflow experiment (`ft-jrn`) as
training/eval, and writes `gguf_eval_results.json`.
```bash
uv run python ml/serve_gguf/eval_gguf.py
```

Note: `ml/serve_gguf/gguf_out/` and the merged adapter directory
(`ml/train/adapters/ft-jrn-merged`) are gitignored — large binaries, regenerable by
re-running this pipeline.

## Results

Quantization (2358 MiB f16 -> 763 MiB Q4_K_M, ~3.1x smaller):

| File | Size |
|---|---|
| `ft-jrn-f16.gguf` | 2364.7 MB |
| `ft-jrn-Q4_K_M.gguf` | 770.3 MB |

30-sample validation set, CPU inference (`n_ctx=2048`, `temperature=0.0`):

| Metric | Unconstrained | Constrained (grammar) |
|---|---|---|
| Valid JSON rate | 1.000 | 1.000 |
| Valence MAE | 0.0693 | 0.0693 |
| Arousal MAE | 0.1053 | 0.1027 |
| Valence corr (r) | 0.8606 | 0.8606 |
| Arousal corr (r) | 0.8768 | 0.8864 |
| Avg latency (ms/request) | 430 | 751 |

On this in-distribution validation set the fine-tuned model already emits valid JSON
reliably even unconstrained (matches the PyTorch-path result in
`ml/train/eval_results.json`), so the grammar doesn't move the valid-JSON-rate needle here.
Its actual value is a **structural guarantee** independent of prompt adherence or
distribution shift — it can't produce malformed output even on adversarial or out-of-
distribution input, at the cost of ~1.7x higher per-request latency from the constrained
token-mask overhead. See `gguf_eval_results.json` for the raw numbers (also logged to the
`ft-jrn` MLflow experiment as run `gguf-eval`).
