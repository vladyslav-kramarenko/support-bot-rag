# llm_loader.py
from langchain_community.llms import LlamaCpp
import os

_llm_instance = None

def get_llm():
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LlamaCpp(
            model_path=os.path.abspath(os.getenv("LLM_MODEL_PATH")),
            temperature=0.1,              # Controls randomness: lower = more deterministic, higher = more creative. 0.1 gives reliable, focused answers
            max_tokens=512,               # Maximum number of tokens the model is allowed to generate in a response
            n_ctx=2048,                   # Size of the context window (max prompt + output tokens). Should match model capabilities
            n_batch=32,                   # Number of tokens to evaluate in parallel (helps speed). Tune based on available RAM/VRAM
            n_threads=8,                  # Number of CPU threads to use for inference. Match to your physical CPU cores (M3 = 8 performance cores)
            repeat_penalty=1.15,          # Penalizes repetition. Values >1.0 discourage repeating tokens. Default is usually 1.1
            repeat_last_n=64,             # How many of the last tokens to apply repeat_penalty to. Higher helps prevent long repetitions
            top_k=40,                     # Restricts token sampling to top_k most likely tokens. Lower = more focused/less diverse output
            top_p=0.9,                    # Nucleus sampling: includes tokens with cumulative probability up to top_p. Controls diversity
            verbose=True                  # Print model loading and inference details (helpful for debugging and performance monitoring)
        )
    return _llm_instance