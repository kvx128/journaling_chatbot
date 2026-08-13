import json
import random
import time
import typer
import numpy as np
import mlflow
from datetime import datetime, timezone
from pathlib import Path
from llama_cpp import Llama, LlamaGrammar

app = typer.Typer(help="Evaluate quantized FT-JRN adapter on validation dataset with llama.cpp")

SYSTEM_PROMPT = (
    "You analyze the emotional content of a short piece of text. Respond with a single JSON "
    "object containing exactly these fields: \"valence\" (float, -1.0 to 1.0, negative to "
    "positive feeling), \"arousal\" (float, -1.0 to 1.0, calm to activated), and "
    "\"emotion_tags\" (a list of short lowercase words naming the emotion, empty list if none "
    "apply). No other text, just the JSON object."
)

@app.command()
def evaluate(
    gguf_path: str = typer.Option("ml/serve_gguf/gguf_out/ft-jrn-Q4_K_M.gguf", help="Path to quantized GGUF model"),
    n_samples: int = typer.Option(30, help="Number of random validation examples to test"),
    seed: int = typer.Option(42, help="Random seed for sampling"),
    grammar_path: str = typer.Option("ml/serve_gguf/grammar.gbnf", help="Path to GBNF grammar file"),
):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("ft-jrn")

    val_file = Path("ml/train/data/journal_val.jsonl")
    if not val_file.exists():
        typer.echo(f"Error: Validation file {val_file} does not exist.", err=True)
        raise typer.Exit(1)

    grammar_file = Path(grammar_path)
    if not grammar_file.exists():
        typer.echo(f"Error: Grammar file {grammar_path} does not exist.", err=True)
        raise typer.Exit(1)

    random.seed(seed)

    typer.echo("Loading validation data...")
    val_examples = []
    with open(val_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            messages = data.get("messages", [])
            if len(messages) >= 3:
                user_text = messages[1]["content"]
                gt_text = messages[2]["content"]
                try:
                    gt_json = json.loads(gt_text)
                    val_examples.append({
                        "user_text": user_text,
                        "gt_json": gt_json
                    })
                except json.JSONDecodeError:
                    continue

    if not val_examples:
        typer.echo("Error: No valid examples found in validation file.", err=True)
        raise typer.Exit(1)

    sampled_examples = random.sample(val_examples, min(n_samples, len(val_examples)))
    actual_samples = len(sampled_examples)

    typer.echo(f"Loading GGUF model from {gguf_path}...")
    llm = Llama(model_path=str(gguf_path), n_ctx=2048, verbose=False)

    typer.echo(f"Loading grammar from {grammar_path}...")
    grammar = LlamaGrammar.from_string(grammar_file.read_text(), verbose=False)

    typer.echo(f"Running inference on {actual_samples} examples...")

    def process_result(result, ex, ex_idx, total_ex, trunc_user, run_type):
        gen_text = result["choices"][0]["message"]["content"].strip()
        gt_json = ex["gt_json"]
        gt_v = float(gt_json.get("valence", 0.0))
        gt_a = float(gt_json.get("arousal", 0.0))

        parsed_ok = False
        pred_v = 0.0
        pred_a = 0.0
        try:
            pred_json = json.loads(gen_text)
            has_keys = all(k in pred_json for k in ["valence", "arousal", "emotion_tags"])
            if has_keys:
                pred_v = float(pred_json["valence"])
                pred_a = float(pred_json["arousal"])
                parsed_ok = True
                print(f"[{ex_idx+1}/{total_ex}] ({run_type}) \"{trunc_user}\" | gt=({gt_v:.2f},{gt_a:.2f}) pred=({pred_v:.2f},{pred_a:.2f})")
            else:
                print(f"[{ex_idx+1}/{total_ex}] ({run_type}) \"{trunc_user}\" | gt=({gt_v:.2f},{gt_a:.2f}) pred=PARSE_FAILED: Missing keys")
        except Exception:
            trunc_gen = gen_text[:60].replace("\n", " ")
            print(f"[{ex_idx+1}/{total_ex}] ({run_type}) \"{trunc_user}\" | gt=({gt_v:.2f},{gt_a:.2f}) pred=PARSE_FAILED: {trunc_gen}")

        return parsed_ok, gt_v, gt_a, pred_v, pred_a

    res = {
        "constrained": {"valid_count": 0, "gt_v": [], "pred_v": [], "gt_a": [], "pred_a": [], "latencies": []},
        "unconstrained": {"valid_count": 0, "gt_v": [], "pred_v": [], "gt_a": [], "pred_a": [], "latencies": []}
    }

    for i, ex in enumerate(sampled_examples):
        user_text = ex["user_text"]
        trunc_user = user_text[:50].replace("\n", " ") + ("..." if len(user_text) > 50 else "")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text}
        ]

        # Unconstrained run
        t0 = time.perf_counter()
        res_un = llm.create_chat_completion(
            messages=messages,
            temperature=0.0,
            max_tokens=80,
        )
        t1 = time.perf_counter()
        res["unconstrained"]["latencies"].append((t1 - t0) * 1000.0)

        ok, gt_v, gt_a, pred_v, pred_a = process_result(res_un, ex, i, actual_samples, trunc_user, "uncons")
        if ok:
            res["unconstrained"]["valid_count"] += 1
            res["unconstrained"]["gt_v"].append(gt_v)
            res["unconstrained"]["pred_v"].append(pred_v)
            res["unconstrained"]["gt_a"].append(gt_a)
            res["unconstrained"]["pred_a"].append(pred_a)

        # Constrained run
        t0 = time.perf_counter()
        res_con = llm.create_chat_completion(
            messages=messages,
            grammar=grammar,
            temperature=0.0,
            max_tokens=80,
        )
        t1 = time.perf_counter()
        res["constrained"]["latencies"].append((t1 - t0) * 1000.0)

        ok, gt_v, gt_a, pred_v, pred_a = process_result(res_con, ex, i, actual_samples, trunc_user, "constr")
        if ok:
            res["constrained"]["valid_count"] += 1
            res["constrained"]["gt_v"].append(gt_v)
            res["constrained"]["pred_v"].append(pred_v)
            res["constrained"]["gt_a"].append(gt_a)
            res["constrained"]["pred_a"].append(pred_a)

    def compute_metrics(d):
        valid_count = d["valid_count"]
        valid_rate = valid_count / actual_samples if actual_samples > 0 else 0.0
        avg_lat = float(np.mean(d["latencies"])) if d["latencies"] else 0.0

        mae_v = None
        mae_a = None
        corr_v = None
        corr_a = None

        if valid_count > 0:
            gt_v_arr = np.array(d["gt_v"])
            pred_v_arr = np.array(d["pred_v"])
            gt_a_arr = np.array(d["gt_a"])
            pred_a_arr = np.array(d["pred_a"])

            mae_v = float(np.mean(np.abs(gt_v_arr - pred_v_arr)))
            mae_a = float(np.mean(np.abs(gt_a_arr - pred_a_arr)))

            c_v = np.corrcoef(gt_v_arr, pred_v_arr)[0, 1] if valid_count > 1 else np.nan
            c_a = np.corrcoef(gt_a_arr, pred_a_arr)[0, 1] if valid_count > 1 else np.nan

            corr_v = float(c_v) if not np.isnan(c_v) else None
            corr_a = float(c_a) if not np.isnan(c_a) else None

        return {
            "valid_json_rate": valid_rate,
            "valence_mae": mae_v,
            "arousal_mae": mae_a,
            "valence_corr": corr_v,
            "arousal_corr": corr_a,
            "avg_latency_ms": avg_lat
        }

    met_un = compute_metrics(res["unconstrained"])
    met_con = compute_metrics(res["constrained"])

    typer.echo("\n--- EVALUATION SUMMARY ---")
    typer.echo(f"{'Metric':<20} | {'Unconstrained':<15} | {'Constrained':<15}")
    typer.echo("-" * 56)

    def fmt(v):
        return f"{v:.4f}" if v is not None else "NaN"

    typer.echo(f"{'Valid JSON Rate':<20} | {met_un['valid_json_rate']:<15.4f} | {met_con['valid_json_rate']:<15.4f}")
    typer.echo(f"{'Valence MAE':<20} | {fmt(met_un['valence_mae']):<15} | {fmt(met_con['valence_mae']):<15}")
    typer.echo(f"{'Arousal MAE':<20} | {fmt(met_un['arousal_mae']):<15} | {fmt(met_con['arousal_mae']):<15}")
    typer.echo(f"{'Valence Corr (r)':<20} | {fmt(met_un['valence_corr']):<15} | {fmt(met_con['valence_corr']):<15}")
    typer.echo(f"{'Arousal Corr (r)':<20} | {fmt(met_un['arousal_corr']):<15} | {fmt(met_con['arousal_corr']):<15}")
    typer.echo(f"{'Avg Latency (ms)':<20} | {met_un['avg_latency_ms']:<15.2f} | {met_con['avg_latency_ms']:<15.2f}")

    with mlflow.start_run(run_name="gguf-eval") as run:
        mlflow.log_params({
            "gguf_path": gguf_path,
            "n_samples": actual_samples,
            "seed": seed
        })

        flat_metrics = {}
        for k, v in met_un.items():
            if v is not None:
                flat_metrics[f"unconstrained_{k}"] = v
        for k, v in met_con.items():
            if v is not None:
                flat_metrics[f"constrained_{k}"] = v

        mlflow.log_metrics(flat_metrics)

        out_file = Path("ml/serve_gguf/gguf_eval_results.json")
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_data = {
            "unconstrained": met_un,
            "constrained": met_con,
            "n_samples": actual_samples,
            "gguf_path": gguf_path,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "mlflow_run_id": run.info.run_id
        }

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2)

        typer.echo(f"\nEvaluation results written to: {out_file}")
        typer.echo(f"MLflow run: {run.info.run_id}")

if __name__ == "__main__":
    app()
