"""
TRL API Version Note:
TRL's SFTTrainer and SFTConfig API have historically experienced frequent changes
(e.g., field renames, argument renames, changes from max_seq_length to max_length,
changes from tokenizer to processing_class). If this script throws an initialization
error on the first real run, the fix is likely a one- or two-argument rename to match
your specific installed TRL/Transformers version, not a logic problem.
"""

import typer
import torch
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

    typer.echo(f"Saving final model adapter and tokenizer to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    typer.echo("Done.")

if __name__ == "__main__":
    app()
