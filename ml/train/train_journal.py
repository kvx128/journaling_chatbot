"""
TRL API Version Note:
TRL's SFTTrainer and SFTConfig API have historically experienced frequent changes
(e.g., field renames, argument renames, changes from max_seq_length to max_length,
changes from tokenizer to processing_class). If this script throws an initialization
error on the first real run, the fix is likely a one- or two-argument rename to match
your specific installed TRL/Transformers version, not a logic problem.
"""

import json
import typer
import torch
import mlflow
import mlflow.pyfunc
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

app = typer.Typer(help="Fine-tune FT-JRN using QLoRA on Llama-3.2-1B-Instruct")

SYSTEM_PROMPT = (
    "You analyze the emotional content of a short piece of text. Respond with a single JSON "
    "object containing exactly these fields: \"valence\" (float, -1.0 to 1.0, negative to "
    "positive feeling), \"arousal\" (float, -1.0 to 1.0, calm to activated), and "
    "\"emotion_tags\" (a list of short lowercase words naming the emotion, empty list if none "
    "apply). No other text, just the JSON object."
)


class FTJrnPredictor(mlflow.pyfunc.PythonModel):
    """Wraps the QLoRA adapter as a loadable/servable MLflow pyfunc model.

    Registering a raw artifact directory (mlflow.log_artifacts) isn't accepted by the
    MLflow 3.x model registry — it requires an actual Logged Model (an MLmodel manifest).
    This wrapper reuses the exact inference pattern already verified in
    services/model_server/inference.py.
    """

    def load_context(self, context):
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel

        with open(context.artifacts["config"]) as f:
            config = json.load(f)
        self.system_prompt = config["system_prompt"]

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForCausalLM.from_pretrained(
            config["base_model"],
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
        )
        self.model = PeftModel.from_pretrained(base, context.artifacts["adapter"])
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(context.artifacts["adapter"])

    def predict(self, context, model_input, params=None):
        texts = model_input["text"].tolist() if hasattr(model_input, "tolist") else list(model_input)
        results = []
        for text in texts:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text},
            ]
            encoded = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
            ).to(self.model.device)
            input_len = encoded["input_ids"].shape[1]
            with torch.no_grad():
                outputs = self.model.generate(
                    **encoded,
                    max_new_tokens=80,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            gen_text = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()
            results.append(gen_text)
        return results

@app.command()
def train(
    model: str = typer.Option("meta-llama/Llama-3.2-1B-Instruct", help="Base model name/path"),
    epochs: float = typer.Option(3.0, help="Number of training epochs"),
    batch_size: int = typer.Option(4, help="Per-device batch size"),
    grad_accum: int = typer.Option(8, help="Gradient accumulation steps"),
    lr: float = typer.Option(2e-4, help="Learning rate"),
    max_seq_length: int = typer.Option(1024, help="Max sequence length"),
    output_dir: str = typer.Option("ml/train/adapters/ft-jrn", help="Output directory"),
):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("ft-jrn")

    train_file = Path("ml/train/data/journal_train.jsonl")
    val_file = Path("ml/train/data/journal_val.jsonl")

    if not train_file.exists():
        typer.echo(f"Error: Training file {train_file} does not exist.", err=True)
        typer.echo("Please run `python -m ml.train.prepare_journal_data` first.", err=True)
        raise typer.Exit(1)

    if not torch.cuda.is_available():
        typer.echo("Warning: CUDA is not available. Training will be extremely slow or fail.", err=True)

    typer.echo("Loading datasets...")
    dataset = load_dataset(
        "json",
        data_files={"train": str(train_file), "eval": str(val_file)}
    )
    train_ds = dataset["train"]
    eval_ds = dataset["eval"]

    steps_per_epoch = max(1, len(train_ds) // (batch_size * grad_accum))
    total_steps = max(1, int(steps_per_epoch * epochs))
    warmup_steps = max(1, round(0.03 * total_steps))
    typer.echo(f"Computed total_steps: {total_steps}, warmup_steps: {warmup_steps}")

    with mlflow.start_run(run_name=f"ft-jrn-{model.split('/')[-1]}") as run:
        typer.echo("Loading tokenizer and configuring QLoRA...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )

        tokenizer = AutoTokenizer.from_pretrained(model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        def to_prompt_completion(example):
            messages = example["messages"]
            prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
            completion = messages[-1]["content"]
            return {"prompt": prompt, "completion": completion}

        train_ds = train_ds.map(to_prompt_completion, remove_columns=["messages"])
        eval_ds = eval_ds.map(to_prompt_completion, remove_columns=["messages"])

        typer.echo("Loading base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            model,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config
        )
        # Required for gradient checkpointing
        base_model.config.use_cache = False

        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
        )

        mlflow.log_params({
            "model": model,
            "epochs": epochs,
            "batch_size": batch_size,
            "grad_accum": grad_accum,
            "lr": lr,
            "max_seq_length": max_seq_length,
            "lora_r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "lora_target_modules": str(lora_config.target_modules),
            "total_steps": total_steps,
            "warmup_steps": warmup_steps,
        })

        # CRITICAL: Only compute loss on the assistant's completion, not the system/user
        # turns. Verified directly against the installed trl==1.9.2: this version has no
        # DataCollatorForCompletionOnlyLM at all (that was a guess against an older/newer
        # API shape and doesn't exist here) — instead SFTConfig has two dedicated fields.
        # `completion_only_loss` is documented as "supported only for prompt-completion
        # datasets". We have converted the conversational format to prompt-completion
        # layout above using the chat template, so completion_only_loss is correct.
        sft_config = SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=lr,
            lr_scheduler_type="cosine",
            warmup_steps=warmup_steps,
            num_train_epochs=epochs,
            bf16=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            packing=False,
            completion_only_loss=True,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="epoch",
            save_total_limit=2,
            report_to="none",
            max_length=max_seq_length,
        )

        typer.echo("Initializing SFTTrainer...")
        trainer = SFTTrainer(
            model=base_model,
            args=sft_config,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            processing_class=tokenizer,
            peft_config=lora_config,
        )

        typer.echo("Starting training...")
        trainer.train()

        for entry in trainer.state.log_history:
            metrics = {k: v for k, v in entry.items() if isinstance(v, (int, float))}
            if metrics:
                mlflow.log_metrics(metrics, step=entry.get("step"))

        typer.echo(f"Saving final model adapter and tokenizer to {output_dir}...")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        config_path = Path(output_dir) / "_ft_jrn_mlflow_config.json"
        config_path.write_text(json.dumps({"base_model": model, "system_prompt": SYSTEM_PROMPT}))

        try:
            mlflow.pyfunc.log_model(
                name="model",
                python_model=FTJrnPredictor(),
                artifacts={"adapter": output_dir, "config": str(config_path)},
                registered_model_name="ft-jrn",
            )
        except Exception as e:
            typer.echo(f"Warning: Failed to log/register model in MLflow Model Registry: {e}", err=True)

        typer.echo("Done.")
        typer.echo(f"MLflow run: {run.info.run_id}")

if __name__ == "__main__":
    app()
