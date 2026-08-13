import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

SYSTEM_PROMPT = (
    "You analyze the emotional content of a short piece of text. Respond with a single JSON "
    "object containing exactly these fields: \"valence\" (float, -1.0 to 1.0, negative to "
    "positive feeling), \"arousal\" (float, -1.0 to 1.0, calm to activated), and "
    "\"emotion_tags\" (a list of short lowercase words naming the emotion, empty list if none "
    "apply). No other text, just the JSON object."
)

_model = None
_tokenizer = None
_load_error = None

def _load_model():
    global _model, _tokenizer
    base_model = os.environ.get("BASE_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
    adapter_dir = os.environ.get("ADAPTER_DIR", "/app/adapter")

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
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)

    _model = model
    _tokenizer = tokenizer

try:
    _load_model()
except Exception as e:
    _load_error = e

def is_ready() -> bool:
    return _load_error is None and _model is not None

def infer_mood(text: str) -> dict:
    if not is_ready():
        return {
            "valence": None,
            "arousal": None,
            "emotion_tags": None,
            "_parse_error": f"Model not loaded: {_load_error}"
        }

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text}
    ]

    encoded = _tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(_model.device)

    input_len = encoded["input_ids"].shape[1]

    with torch.no_grad():
        outputs = _model.generate(
            **encoded,
            max_new_tokens=80,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=_tokenizer.pad_token_id
        )

    gen_tokens = outputs[0][input_len:]
    gen_text = _tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

    try:
        return json.loads(gen_text)
    except Exception:
        return {
            "valence": None,
            "arousal": None,
            "emotion_tags": None,
            "_parse_error": gen_text[:200]
        }
