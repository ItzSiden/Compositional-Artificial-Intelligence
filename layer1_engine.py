"""
LAYER 1: Small Language Core (Inference Engine)
------------------------------------------------
Loads and runs the quantized LLM using llama.cpp.
Context window locked to 2048 to prevent RAM overflow.
CPU-only, no GPU required.
"""

import os
from llama_cpp import Llama

MODEL_DIR = "models"
MODEL_FILE = "Llama-3.2-1B-Instruct-Uncensored.IQ3_M.gguf"

class LLMEngine:
    def __init__(self):
        self.model = None
        self._load_model()

    def _load_model(self):
        model_path = os.path.join(MODEL_DIR, MODEL_FILE)
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at '{model_path}'.\n"
                f"Run setup.sh first, or manually download from:\n"
                f"https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF"
            )
        print(f"[Layer 1] Loading model from {model_path} ...")
        self.model = Llama(
            model_path=model_path,
            n_ctx=2048,        # Strict context window limit to prevent RAM explosion
            n_threads=4,       # CPU threads
            n_gpu_layers=0,    # CPU only
            verbose=False
        )
        print("[Layer 1] Model loaded successfully.")

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2, stream: bool = True):
        """Stream or return the LLM output for the given prompt."""
        if stream:
            output = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=True,
                echo=False,
                stop=["<|eot_id|>", "[INST]", "User:", "You:"]
            )
            for chunk in output:
                token = chunk["choices"][0]["text"]
                yield token
        else:
            output = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                echo=False,
                stop=["<|eot_id|>", "[INST]", "User:", "You:"]
            )
            yield output["choices"][0]["text"]
