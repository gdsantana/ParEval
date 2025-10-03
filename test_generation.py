#!/usr/bin/env python3
import json
import torch
from transformers import pipeline

model_path = "deepseek-ai/deepseek-coder-1.3b-base"

simple_prompt = """void simple_add(int* a, int* b, int* c, int n) {
    for (int i = 0; i < n; i++) {"""

print(f"Testing model: {model_path}")
print(f"Prompt:\n{simple_prompt}\n")

generator = pipeline(
    task="text-generation", 
    model=model_path, 
    torch_dtype=torch.bfloat16,
    device=0 if torch.cuda.is_available() else -1,
)

tokenizer = generator.tokenizer
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "left"

print(f"\nTokenizer info:")
print(f"  eos_token_id: {tokenizer.eos_token_id}")
print(f"  eos_token: {tokenizer.eos_token}")
print(f"  pad_token_id: {tokenizer.pad_token_id}")

print("\n" + "="*60)
print("Test 1: Default generation (do_sample=False, temperature=0.2)")
print("="*60)
output1 = generator(
    simple_prompt,
    max_new_tokens=100,
    do_sample=False,
    temperature=0.2,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
print(f"Output:\n{output1[0]['generated_text']}\n")

print("\n" + "="*60)
print("Test 2: With sampling enabled (do_sample=True, temperature=0.2)")
print("="*60)
output2 = generator(
    simple_prompt,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.2,
    top_p=0.95,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
print(f"Output:\n{output2[0]['generated_text']}\n")

print("\n" + "="*60)
print("Test 3: With higher temperature (do_sample=True, temperature=0.8)")
print("="*60)
output3 = generator(
    simple_prompt,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.8,
    top_p=0.95,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)
print(f"Output:\n{output3[0]['generated_text']}\n")

print("\n" + "="*60)
print("Test 4: With eos_token_id=None")
print("="*60)
output4 = generator(
    simple_prompt,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.2,
    top_p=0.95,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=None,
)
print(f"Output:\n{output4[0]['generated_text']}\n")
