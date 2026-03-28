# inference.py
# IndiaMed42 — India's First Hindi Medical AI

import torch
from transformers import AutoTokenizer
from peft import PeftModel
from transformers import AutoModelForCausalLM

def load_model():
    print("Loading IndiaMed42...")
    
    base_model = AutoModelForCausalLM.from_pretrained(
        "m42-health/Llama3-Med42-8B",
        torch_dtype=torch.float16,
        device_map="auto",
        load_in_4bit=True
    )
    
    model = PeftModel.from_pretrained(
        base_model,
        "mazzu43/IndiaMed42"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        "mazzu43/IndiaMed42"
    )
    
    print("✅ IndiaMed42 Ready!")
    return model, tokenizer

def ask(question, language="hindi"):
    
    if language == "hindi":
        system = "Tu IndiaMed42 hai — India ka AI medical assistant. Hindi mein jawab de."
    else:
        system = "You are IndiaMed42 — India's AI medical assistant."
    
    text = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system}<|eot_id|><|start_header_id|>user<|end_header_id|>
{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
    
    inputs = tokenizer(
        text, return_tensors="pt"
    ).to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    return response

# Test karo
if __name__ == "__main__":
    model, tokenizer = load_model()
    
    # Hindi test
    print("\n--- Hindi Test ---")
    r = ask("Mujhe bukhar hai 102F. Kya karoon?")
    print(r)
    
    # English test
    print("\n--- English Test ---")
    r = ask(
        "Patient has fever 102F and headache.",
        language="english"
    )
    print(r)
