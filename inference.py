# inference.py
# IndiaMed42 — India's First Hindi Medical AI

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# System prompts as module-level constants to avoid duplication
SYSTEM_PROMPTS = {
    "hindi": "Tu IndiaMed42 hai — India ka AI medical assistant. Hindi mein jawab de.",
    "english": "You are IndiaMed42 — India's AI medical assistant.",
}

def load_model():
    print("Loading IndiaMed42...")

    try:
        # Use BitsAndBytesConfig for proper 4-bit quantization instead of
        # the deprecated load_in_4bit + torch_dtype combination, which can
        # conflict and produce suboptimal memory layout.
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            "m42-health/Llama3-Med42-8B",
            quantization_config=bnb_config,
            device_map="auto",
        )

        model = PeftModel.from_pretrained(
            base_model,
            "mazzu43/IndiaMed42"
        )

        tokenizer = AutoTokenizer.from_pretrained(
            "mazzu43/IndiaMed42"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load IndiaMed42: {e}") from e

    # Ensure a pad token is always set (avoids silent batching issues)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Switch to eval mode so dropout layers are disabled during inference
    model.eval()

    print("✅ IndiaMed42 Ready!")
    return model, tokenizer

def ask(question, model, tokenizer, language="hindi", max_new_tokens=200):
    if not question or not isinstance(question, str):
        raise ValueError("question must be a non-empty string")

    system = SYSTEM_PROMPTS.get(language, SYSTEM_PROMPTS["english"])

    text = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system}<|eot_id|><|start_header_id|>user<|end_header_id|>
{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""

    # Derive target device from the model instead of hard-coding "cuda",
    # so the code works on CPU and MPS as well.
    device = next(model.parameters()).device
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # torch.inference_mode() is a stricter, slightly faster alternative to
    # torch.no_grad() for pure inference: it also disables version tracking.
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    input_length = inputs["input_ids"].shape[1]
    response = tokenizer.decode(
        outputs[0, input_length:],
        skip_special_tokens=True
    )
    return response

# Test karo
if __name__ == "__main__":
    model, tokenizer = load_model()

    # Hindi test
    print("\n--- Hindi Test ---")
    r = ask("Mujhe bukhar hai 102F. Kya karoon?", model, tokenizer)
    print(r)

    # English test
    print("\n--- English Test ---")
    r = ask(
        "Patient has fever 102F and headache.",
        model,
        tokenizer,
        language="english"
    )
    print(r)
