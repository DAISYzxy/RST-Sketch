# -*- coding: utf-8 -*-
"""
Batch two-turn dialogue summarization (JSON → JSON version)
- Reads a single JSON file `wcep_selected_edus.json`, each item contains "selected_text".
- Runs summarization for each item.
- Appends all summaries into one list and saves as a single JSON file `summaries.json`.

Usage: python summarize_all_files.py
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import sys
import json

# ----------------- 配置区（按需修改） -----------------
model_path = "/export/data/RA_Work/seraveea/Llama-2-7b-chat-hf"
input_json = "wcep_selected_edus_0dot7.json"
output_json = "wcep_0dot7_summaries.json"
max_new_tokens = 320
device = "cuda" if torch.cuda.is_available() else "cpu"
safety_margin = 50
# ------------------------------------------------------

print("Device:", device)
print("Loading tokenizer and model from:", model_path)
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        device_map="auto" if device.startswith("cuda") else None,
    )
except Exception as e:
    print("Failed to load model/tokenizer:", e)
    sys.exit(1)

# Summarization instruction
instruction = (
    "Summarize the meeting transcript that will follow. "
    "Produce a single concise summary in plain English with no quotations or verbatim transcript. "
    "Keep it under 300 words. Output only the summary (no headers, no bullet numbers)."
)
system_message = "You are a concise and factual summarizer."

def build_prompt(transcript: str) -> str:
    return (
        "<s>[INST] <<SYS>>\n"
        + system_message
        + "\n<</SYS>>\n\n"
        + instruction
        + "\n\n[TRANSCRIPT START]\n"
        + transcript
        + "\n[TRANSCRIPT END]\n[/INST]"
    )

# Context window
model_max = getattr(tokenizer, "model_max_length", None)
if model_max is None or model_max > 100000:
    model_max = 4096

# ---------- 读取输入 JSON ----------
if not os.path.exists(input_json):
    print(f"Input JSON not found: {input_json}")
    sys.exit(0)

try:
    with open(input_json, "r", encoding="utf-8") as jf:
        data = json.load(jf)
except Exception as e:
    print(f"Failed to load JSON: {e}")
    sys.exit(1)

if not isinstance(data, list) or not data:
    print("JSON content is empty or invalid.")
    sys.exit(0)

print(f"Loaded {len(data)} items from {input_json}")

results = []

# ---------- 处理每个 item ----------
for idx, item in enumerate(data):
    if not isinstance(item, dict):
        print(f"\nSkipping index {idx}: not a dict")
        continue

    transcript = (item.get("selected_text") or "").strip()
    if not transcript:
        print(f"\nSkipping index {idx}: empty selected_text")
        continue

    # 推断 item id
    item_id = item.get("id") or item.get("filename") or f"item_{idx:05d}"

    print(f"\n--- Processing item {idx} ({item_id}) ---")
    print("TRANSCRIPT length (chars):", len(transcript))

    prompt = build_prompt(transcript)
    try:
        encoded = tokenizer(prompt, return_tensors="pt", truncation=False, add_special_tokens=False)
        input_token_count = encoded["input_ids"].size(1)
    except Exception:
        tr_tokens = tokenizer(transcript, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        prefix_tokens = tokenizer(build_prompt(""), return_tensors="pt", add_special_tokens=False)["input_ids"].size(1)
        input_token_count = prefix_tokens + tr_tokens.size(0)

    print("Token counts (approx):", input_token_count)
    if input_token_count + max_new_tokens + safety_margin > model_max:
        allowed_input_tokens = model_max - max_new_tokens - safety_margin
        tr_tokens = tokenizer(transcript, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
        if tr_tokens.size(0) > allowed_input_tokens:
            kept = tr_tokens[-allowed_input_tokens:]
            transcript = tokenizer.decode(kept, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            print(f"Truncated transcript to last {allowed_input_tokens} tokens.")
            prompt = build_prompt(transcript)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=False).to(model.device)
    gen_kwargs = dict(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    print("Generating summary...")
    try:
        outputs = model.generate(**gen_kwargs)
        out_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        print(f"Generation failed for {item_id}: {e}")
        continue

    marker = "[/INST]"
    if marker in out_text:
        summary = out_text.split(marker)[-1].strip()
    elif out_text.startswith(prompt):
        summary = out_text[len(prompt):].strip()
    else:
        summary = out_text.strip()

    summary = "\n".join([line.strip() for line in summary.splitlines() if line.strip()])

    results.append({
        "index": idx,
        "id": item_id,
        "summary": summary
    })

# ---------- 保存结果为一个 JSON ----------
try:
    with open(output_json, "w", encoding="utf-8") as jf:
        json.dump(results, jf, ensure_ascii=False, indent=2)
    print(f"\n✅ Saved all {len(results)} summaries to {output_json}")
except Exception as e:
    print(f"Failed to save results: {e}")

print("\nAll done.")
