import typer
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

app = typer.Typer(help="Merge FT-JRN adapter into base model")

@app.command()
def merge(
    adapter_dir: str = typer.Option("ml/train/adapters/ft-jrn", help="Path to trained adapter"),
    base_model: str = typer.Option("meta-llama/Llama-3.2-1B-Instruct", help="Base model name/path"),
    output_dir: str = typer.Option("ml/train/adapters/ft-jrn-merged", help="Output directory"),
):
    if not Path(adapter_dir).exists():
        typer.echo(f"Error: Adapter directory {adapter_dir} does not exist.", err=True)
        raise typer.Exit(1)

    typer.echo("Loading base model in bf16 on CPU...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu"
    )

    typer.echo("Loading PEFT adapter...")
    model = PeftModel.from_pretrained(base, adapter_dir)

    typer.echo("Merging adapter into base weights...")
    merged_model = model.merge_and_unload()

    typer.echo("Loading tokenizer from adapter dir...")
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)

    typer.echo(f"Saving merged model and tokenizer to {output_dir}...")
    merged_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    typer.echo("Done.")

if __name__ == "__main__":
    app()
