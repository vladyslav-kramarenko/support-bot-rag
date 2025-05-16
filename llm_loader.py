# llm_loader.py
from langchain_community.llms import LlamaCpp
import os
import yaml

_llm_instance = None

# === Load and parse config.yaml ===
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

model_profiles = config.get("model_profiles", {})
active_profile = config.get("model_profile")

if not active_profile:
    raise ValueError("❌ 'model_profile' is not defined in config.yaml")

if active_profile not in model_profiles:
    raise ValueError(f"❌ Model profile '{active_profile}' not found in config.yaml")

model_config = model_profiles[active_profile]

def get_model_config():
    return model_config

# === Load the LLM only once ===
def get_llm():
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LlamaCpp(
            model_path=os.path.abspath(model_config.get("path")),
            temperature=model_config.get("temperature", 0.1),         # Controls randomness (lower = more focused)
            max_tokens=model_config.get("max_tokens", 512),           # Max tokens to generate in the response
            n_ctx=model_config.get("n_ctx", 4096),                    # Context window size (input + output tokens)
            n_batch=model_config.get("n_batch", 32),                  # Tokens to process in parallel (speed/memory)
            n_threads=model_config.get("n_threads", 8),               # CPU threads to use (match physical cores)
            repeat_penalty=model_config.get("repeat_penalty", 1.15), # Discourages repetition (>1 = stronger penalty)
            repeat_last_n=model_config.get("repeat_last_n", 64),     # How many recent tokens penalty applies to
            top_k=model_config.get("top_k", 40),                      # Sample only from top_k tokens (focuses output)
            top_p=model_config.get("top_p", 0.9),                     # Nucleus sampling threshold (controls diversity)
            verbose=True                                              # Logs model details (load + generation)
        )
    return _llm_instance