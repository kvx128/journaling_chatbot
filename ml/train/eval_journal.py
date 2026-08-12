import json
import random
import typer
import torch
import numpy as np
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
    valid_rate = (valid_count / actual_samples) * 100 if actual_samples > 0 else 0.0
    typer.echo(f"Valid JSON Rate : {valid_rate:.1f}% ({valid_count}/{actual_samples})")

    if valid_count > 0:
        gt_v_arr = np.array(gt_valence_list)
        pred_v_arr = np.array(pred_valence_list)
        gt_a_arr = np.array(gt_arousal_list)
        pred_a_arr = np.array(pred_arousal_list)

        mae_v = np.mean(np.abs(gt_v_arr - pred_v_arr))
        mae_a = np.mean(np.abs(gt_a_arr - pred_a_arr))

        corr_v = np.corrcoef(gt_v_arr, pred_v_arr)[0, 1] if valid_count > 1 else np.nan
        corr_a = np.corrcoef(gt_a_arr, pred_a_arr)[0, 1] if valid_count > 1 else np.nan

        typer.echo(f"Valence MAE     : {mae_v:.4f}")
        typer.echo(f"Arousal MAE     : {mae_a:.4f}")
        typer.echo(f"Valence Corr (r): {corr_v:.4f}")
        typer.echo(f"Arousal Corr (r): {corr_a:.4f}")
    else:
        typer.echo("No valid JSON outputs to compute metrics.")

if __name__ == "__main__":
    app()
