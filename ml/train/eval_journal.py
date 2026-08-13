import json
import random
import typer
import torch
import numpy as np
import mlflow
from datetime import datetime, timezone
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel

app = typer.Typer(help="Evaluate FT-JRN adapter on validation dataset")

SYSTEM_PROMPT = (
    "You analyze the emotional content of a short piece of text. Respond with a single JSON "
    "object containing exactly these fields: \"valence\" (float, -1.0 to 1.0, negative to "
    "positive feeling), \"arousal\" (float, -1.0 to 1.0, calm to activated), and "
    "\"emotion_tags\" (a list of short lowercase words naming the emotion, empty list if none "
    "apply). No other text, just the JSON object."
)

@app.command()
def evaluate(
    adapter_dir: str = typer.Option("ml/train/adapters/ft-jrn", help="Path to trained adapter"),
    base_model: str = typer.Option("meta-llama/Llama-3.2-1B-Instruct", help="Base model name/path"),
    n_samples: int = typer.Option(30, help="Number of random validation examples to test"),
    seed: int = typer.Option(42, help="Random seed for sampling"),
):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("ft-jrn")

    val_file = Path("ml/train/data/journal_val.jsonl")
    if not val_file.exists():
        typer.echo(f"Error: Validation file {val_file} does not exist.", err=True)
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

    with mlflow.start_run(run_name="eval") as run:
        mlflow.log_params({
            "adapter_dir": adapter_dir,
            "base_model": base_model,
            "n_samples": actual_samples,
            "seed": seed
        })

        typer.echo("Loading base model in 4-bit...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config
        )

        typer.echo(f"Loading adapter from {adapter_dir}...")
        model = PeftModel.from_pretrained(model, adapter_dir)
        model.eval()

        typer.echo(f"Loading tokenizer from {adapter_dir}...")
        tokenizer = AutoTokenizer.from_pretrained(adapter_dir)

        typer.echo(f"Running inference on {actual_samples} examples...")

        valid_count = 0
        gt_valence_list = []
        pred_valence_list = []
        gt_arousal_list = []
        pred_arousal_list = []

        for i, ex in enumerate(sampled_examples):
            user_text = ex["user_text"]
            gt_json = ex["gt_json"]

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_text}
            ]

            encoded = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            ).to(model.device)

            input_len = encoded["input_ids"].shape[1]

            with torch.no_grad():
                outputs = model.generate(
                    **encoded,
                    max_new_tokens=80,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=tokenizer.pad_token_id
                )

            gen_tokens = outputs[0][input_len:]
            gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

            trunc_user = user_text[:50].replace("\n", " ")
            if len(user_text) > 50:
                trunc_user += "..."

            gt_v = float(gt_json.get("valence", 0.0))
            gt_a = float(gt_json.get("arousal", 0.0))

            try:
                pred_json = json.loads(gen_text)
                has_keys = all(k in pred_json for k in ["valence", "arousal", "emotion_tags"])
                if has_keys:
                    pred_v = float(pred_json["valence"])
                    pred_a = float(pred_json["arousal"])

                    valid_count += 1
                    gt_valence_list.append(gt_v)
                    pred_valence_list.append(pred_v)
                    gt_arousal_list.append(gt_a)
                    pred_arousal_list.append(pred_a)

                    print(f"[{i+1}/{actual_samples}] \"{trunc_user}\" | gt=({gt_v:.2f},{gt_a:.2f}) pred=({pred_v:.2f},{pred_a:.2f})")
                else:
                    missing_keys = [k for k in ["valence", "arousal", "emotion_tags"] if k not in pred_json]
                    print(f"[{i+1}/{actual_samples}] \"{trunc_user}\" | gt=({gt_v:.2f},{gt_a:.2f}) pred=PARSE_FAILED: Missing required keys {missing_keys}")
            except Exception:
                trunc_gen = gen_text[:60].replace("\n", " ")
                print(f"[{i+1}/{actual_samples}] \"{trunc_user}\" | gt=({gt_v:.2f},{gt_a:.2f}) pred=PARSE_FAILED: {trunc_gen}")

        typer.echo("\n--- EVALUATION SUMMARY ---")
        valid_rate_frac = valid_count / actual_samples if actual_samples > 0 else 0.0
        valid_rate_pct = valid_rate_frac * 100
        typer.echo(f"Valid JSON Rate : {valid_rate_pct:.1f}% ({valid_count}/{actual_samples})")

        mae_v = None
        mae_a = None
        corr_v = None
        corr_a = None

        if valid_count > 0:
            gt_v_arr = np.array(gt_valence_list)
            pred_v_arr = np.array(pred_valence_list)
            gt_a_arr = np.array(gt_arousal_list)
            pred_a_arr = np.array(pred_arousal_list)

            mae_v = float(np.mean(np.abs(gt_v_arr - pred_v_arr)))
            mae_a = float(np.mean(np.abs(gt_a_arr - pred_a_arr)))

            c_v = np.corrcoef(gt_v_arr, pred_v_arr)[0, 1] if valid_count > 1 else np.nan
            c_a = np.corrcoef(gt_a_arr, pred_a_arr)[0, 1] if valid_count > 1 else np.nan

            corr_v = float(c_v) if not np.isnan(c_v) else None
            corr_a = float(c_a) if not np.isnan(c_a) else None

            metrics_to_log = {
                "valid_json_rate": valid_rate_frac,
                "valence_mae": mae_v,
                "arousal_mae": mae_a,
                "valence_corr": corr_v,
                "arousal_corr": corr_a
            }

            mlflow.log_metrics({k: v for k, v in metrics_to_log.items() if v is not None})

            typer.echo(f"Valence MAE     : {mae_v:.4f}")
            typer.echo(f"Arousal MAE     : {mae_a:.4f}")

            cv_disp = f"{corr_v:.4f}" if corr_v is not None else "NaN"
            ca_disp = f"{corr_a:.4f}" if corr_a is not None else "NaN"

            typer.echo(f"Valence Corr (r): {cv_disp}")
            typer.echo(f"Arousal Corr (r): {ca_disp}")
        else:
            typer.echo("No valid JSON outputs to compute metrics.")
            mlflow.log_metrics({"valid_json_rate": valid_rate_frac})

        eval_results_file = Path("ml/train/eval_results.json")
        eval_results_file.parent.mkdir(parents=True, exist_ok=True)

        summary_dict = {
            "valid_json_rate": valid_rate_frac,
            "valence_mae": mae_v,
            "arousal_mae": mae_a,
            "valence_corr": corr_v,
            "arousal_corr": corr_a,
            "n_samples": actual_samples,
            "adapter_dir": adapter_dir,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "mlflow_run_id": run.info.run_id
        }

        with open(eval_results_file, "w", encoding="utf-8") as f:
            json.dump(summary_dict, f, indent=2)

        typer.echo(f"Evaluation results written to: {eval_results_file}")
        typer.echo(f"MLflow run: {run.info.run_id}")

if __name__ == "__main__":
    app()
