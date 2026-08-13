import os
import sys
import subprocess
import ctypes
import typer
from pathlib import Path

app = typer.Typer(help="Quantize merged model to GGUF Q4_K_M")

@app.command()
def quantize(
    merged_dir: str = typer.Option("ml/train/adapters/ft-jrn-merged", help="Path to merged model"),
    output_dir: str = typer.Option("ml/serve_gguf/gguf_out", help="Output directory"),
):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    f16_path = out_dir / "ft-jrn-f16.gguf"
    q4_path = out_dir / "ft-jrn-Q4_K_M.gguf"

    typer.echo("Step A: Converting HF model to f16 GGUF...")
    env = os.environ.copy()
    env["NO_LOCAL_GGUF"] = "1"

    convert_script = "ml/serve_gguf/vendor/llama_cpp_convert/convert_hf_to_gguf.py"
    cmd = [
        sys.executable,
        convert_script,
        merged_dir,
        "--outtype", "f16",
        "--outfile", str(f16_path)
    ]

    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if res.returncode != 0:
        typer.echo("Error converting to GGUF:", err=True)
        typer.echo(res.stderr, err=True)
        raise typer.Exit(1)

    typer.echo(f"Step A complete. Created {f16_path}")

    typer.echo("Step B: Quantizing to Q4_K_M...")
    import llama_cpp

    params = llama_cpp.llama_model_quantize_default_params()
    params.ftype = llama_cpp.LLAMA_FTYPE_MOSTLY_Q4_K_M

    rc = llama_cpp.llama_model_quantize(
        str(f16_path).encode("utf-8"),
        str(q4_path).encode("utf-8"),
        ctypes.byref(params)
    )

    if rc != 0:
        typer.echo(f"Error quantizing model. Return code: {rc}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Step B complete. Created {q4_path}")

    f16_size = os.path.getsize(f16_path) / (1024 * 1024)
    q4_size = os.path.getsize(q4_path) / (1024 * 1024)

    typer.echo(f"File sizes:")
    typer.echo(f"  f16 GGUF:    {f16_size:.2f} MB")
    typer.echo(f"  Q4_K_M GGUF: {q4_size:.2f} MB")
    typer.echo("Output paths:")
    typer.echo(f"  {f16_path}")
    typer.echo(f"  {q4_path}")
    typer.echo("Done.")

if __name__ == "__main__":
    app()
