# Vendored from ggml-org/llama.cpp

`convert_hf_to_gguf.py`, `convert_lora_to_gguf.py`, and `conversion/` are copied verbatim
from https://github.com/ggml-org/llama.cpp at commit `e79e4bf660e19f2ad851e06c6913f7a8c5852621`
(MIT licensed, see `LICENSE`). Not published on PyPI, so this is the standard way projects
depend on it.

Run with `NO_LOCAL_GGUF=1` set so it imports the pip-installed `gguf` package (see the `ml`
extra in `pyproject.toml`) instead of expecting a sibling `gguf-py/` checkout.
