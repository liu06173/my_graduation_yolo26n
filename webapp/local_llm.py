#!/usr/bin/env python3
"""本地Qwen3-4B推理服务 (CPU模式，不影响训练)"""
import sys, os, json, time, threading
from flask import Flask, request, jsonify, Response, stream_with_context

MODEL_PATH = r"D:\AI\Qwen3___5-4B"

app = Flask(__name__)
model = None
tokenizer = None
loaded = False
lock = threading.Lock()

def load_model():
    global model, tokenizer, loaded
    print(f"Loading Qwen3-4B from {MODEL_PATH}...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype="auto",
        device_map="cpu",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    loaded = True
    print("Qwen3-4B loaded on CPU", flush=True)

@app.route("/v1/chat/completions", methods=["POST"])
def chat():
    if not loaded:
        return jsonify({"error": "Model loading..."}), 503

    data = request.json
    messages = data.get("messages", [])
    max_tokens = min(data.get("max_tokens", 1024), 2048)
    temperature = data.get("temperature", 0.7)

    # Build prompt from messages
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    with lock:
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
        response = tokenizer.decode(
            outputs[0][len(inputs.input_ids[0]):],
            skip_special_tokens=True,
        )

    return jsonify({
        "choices": [{
            "message": {
                "role": "assistant",
                "content": response.strip(),
            }
        }]
    })

@app.route("/v1/models", methods=["GET"])
def models():
    return jsonify({
        "data": [{"id": "qwen3-4b-local", "object": "model"}]
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ready" if loaded else "loading"})

if __name__ == "__main__":
    threading.Thread(target=load_model, daemon=True).start()
    app.run(host="127.0.0.1", port=5801, threaded=True)
