import os
import torch
import numpy as np
import pandas as pd
from transformer_lens import HookedTransformer
from sae_lens import SAE

# =========================
# global config
# =========================
OUTPUT_ROOT = "./sae_data"

DATA_PATH_E = "./EEMO.csv"
DATA_PATH_C = "./CEMO.csv"

# -------------------------
# gemma: 0.25/0.5/0.75 -> layer 9/20/31
#       1/2 -> width 16k/131k
# -------------------------
GEMMA_SAE_MAP = {
    "0.25_1": "layer_9/width_16k/average_l0_88",
    "0.25_2": "layer_9/width_131k/average_l0_121",

    "0.5_1":  "layer_20/width_16k/average_l0_91",
    "0.5_2":  "layer_20/width_131k/average_l0_81",

    "0.75_1": "layer_31/width_16k/average_l0_76",
    "0.75_2": "layer_31/width_131k/average_l0_109",
}

# -------------------------
# llama: 0.25/0.5/0.75 -> l7/l15/l23
#        1/2 -> 8x/32x
# release: llama_scope_lxr_{8x or 32x}
# -------------------------
LLAMA_SAE_MAP = {
    "0.25_1": ("llama_scope_lxr_8x",  "l7r_8x"),
    "0.25_2": ("llama_scope_lxr_32x", "l7r_32x"),

    "0.5_1":  ("llama_scope_lxr_8x",  "l15r_8x"),
    "0.5_2":  ("llama_scope_lxr_32x", "l15r_32x"),

    "0.75_1": ("llama_scope_lxr_8x",  "l23r_8x"),
    "0.75_2": ("llama_scope_lxr_32x", "l23r_32x"),
}

CONFIG = {
    "gemma": {
        "hf_model_name": "google/gemma-2-9b-it",
        "sae_release": "gemma-scope-9b-it-res",
        "sae_map": GEMMA_SAE_MAP,  # sae_key -> sae_id(str)
    },
    "llama": {
        "hf_model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
        "sae_release": None,      # 对 llama 来说 release 会随 sae_key 变化
        "sae_map": LLAMA_SAE_MAP, # sae_key -> (release, sae_id)
    },
}

# =========================
# data reader
# =========================
def load_word_list(csv_path: str, word_col: str = "Word"):
    df = pd.read_csv(csv_path)
    return df[word_col].astype(str).to_list()

# =========================
# SAE feature extract & aggregate
# =========================
@torch.no_grad()
def extract_word_sae_feature(model, sae, hook_point: str, word: str):
    """
    Given a word, extract its SAE feature from the specified hook_point, and aggregate to a single vector.
    """
    _, cache = model.run_with_cache(word, prepend_bos=True, names_filter=[hook_point])
    feature_acts = sae.encode(cache[hook_point])      # [1, seq, n_features]
    vec = feature_acts[0, 1:, :].sum(dim=0)           # [n_features]
    return vec.float().cpu().numpy()

def get_word_list_feature(model, sae, hook_point: str, word_list, log_every: int = 1000):
    feats = []
    for i, w in enumerate(word_list):
        if log_every and i > 0 and i % log_every == 0:
            print(f"[progress] {i}/{len(word_list)}")
        feats.append(extract_word_sae_feature(model, sae, hook_point, w))
    return np.stack(feats, axis=0)  # [N, n_features]

# =========================
# load model+SAE+extract+save
# =========================
def run_extract(
    model_name: str,
    sae_key: str,
    device: str = "cuda",
    word_limit: int = None,
):
    if model_name not in CONFIG:
        raise ValueError(f"Unknown model_name={model_name}. Choices: {list(CONFIG.keys())}")

    model_cfg = CONFIG[model_name]
    if sae_key not in model_cfg["sae_map"]:
        raise ValueError(
            f"Unknown sae_key={sae_key} for model_name={model_name}. "
            f"Choices: {list(model_cfg['sae_map'].keys())}"
        )

    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"[device] {torch_device}")

    # 1) load HookedTransformer
    print(f"[load] HookedTransformer: {model_cfg['hf_model_name']}")
    model = HookedTransformer.from_pretrained(
        model_name=model_cfg["hf_model_name"],
        device=torch_device
    )

    # 2) load SAE，get hook_point
    if model_name == "gemma":
        sae_release = model_cfg["sae_release"]
        sae_id = model_cfg["sae_map"][sae_key]
        print(f"[load] SAE: release={sae_release}, sae_id={sae_id}")
        sae, _, _ = SAE.from_pretrained(release=sae_release, sae_id=sae_id, device=torch_device)
    else:
        sae_release, sae_id = model_cfg["sae_map"][sae_key]
        print(f"[load] SAE: release={sae_release}, sae_id={sae_id}")

        sae, _, _ = SAE.from_pretrained(sae_release, sae_id, device=torch_device)

    sae = sae.to(torch_device)
    hook_point = sae.cfg.hook_name
    print(f"[hook_point] {hook_point}")

    # 3) load word list
    words_e = load_word_list(DATA_PATH_E, "Word")
    words_c = load_word_list(DATA_PATH_C, "Word")

    if word_limit is not None:
        words_e = words_e[:word_limit]
        words_c = words_c[:word_limit]

    # 4) extract features
    print(f"[extract] EEMO words: {len(words_e)}")
    sae_acts_e = get_word_list_feature(model, sae, hook_point, words_e, log_every=1000)

    print(f"[extract] CEMO words: {len(words_c)}")
    sae_acts_c = get_word_list_feature(model, sae, hook_point, words_c, log_every=1000)

    # 5) save path
    out_dir = os.path.join(OUTPUT_ROOT, f"sae_features_{model_name}", sae_key)
    os.makedirs(out_dir, exist_ok=True)

    path_e = os.path.join(out_dir, "sae_acts_e.npy")
    path_c = os.path.join(out_dir, "sae_acts_c.npy")

    np.save(path_e, sae_acts_e)
    np.save(path_c, sae_acts_c)

    print("[done]")
    print(path_e)
    print(path_c)


if __name__ == "__main__":
    # eg 1：gemma -> layer_31 / width_131k => 0.75_2
    run_extract(model_name="gemma", sae_key="0.75_2", device="cuda")

    # eg 2：llama -> l23 / 32x => 0.75_2
    # run_extract(model_name="llama", sae_key="0.75_2", device="cuda")