import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import copy
import json
import random

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import NMF
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer
from sae_lens import SAE


# =========================
# Global Config
# =========================
MODEL_ID = "google/gemma-2-9b-it"
RELEASE = "gemma-scope-9b-it-res"
SAE_ID = "layer_9/width_16k/average_l0_88"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32

SEED = 42

# ====== 运行模式 ======
# "erasure"   -> 概念擦除
RUN_MODE = "erasure"

# ====== 擦除配置 ======
# edit_mode:
#   "replace"  : h' = Dec(z')
#   "delta"    : h' = h + alpha * (Dec(z') - Dec(z))   [推荐]
#   "subtract" : h' = h - alpha * sum_f z_f d_f
EDIT_MODE = "delta"

# intervene_on:
#   "generation" : 只在生成阶段干预（推荐）
#   "last_prompt": 只干预 prompt 最后一个 token
#   "prompt_tail": 干预 prompt 最后 20%
#   "all"        : 干预当前 forward 的所有 token
INTERVENE_ON = "all"

# 概念增强/擦除强度
ALPHA_LIST = [0.0,1.0]

# target emo  id
# 4: anger
# 12: disgust
# 16: fear
# 19: joy
# 23: sadness
# 26: surprise

EMO_ID = 12



# 输出句子数
K_WORDS = 410

# 输入 CSV 路径
CSV_PATH = "./EEMO.csv"

# 输出路径
OUTPUT_JSON = f"./gemma_results/{RUN_MODE}_emo_{EMO_ID}_k_{K_WORDS}_{EDIT_MODE}_{INTERVENE_ON}.json"


# =========================
# Utils
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_terminators(tokenizer):
    terms = []
    if tokenizer.eos_token_id is not None:
        terms.append(tokenizer.eos_token_id)

    special_tokens = ["<|eot_id|>", "<end_of_turn>"]
    for tok in special_tokens:
        try:
            tok_id = tokenizer.convert_tokens_to_ids(tok)
            if tok_id is not None and tok_id != tokenizer.unk_token_id:
                terms.append(tok_id)
        except Exception:
            pass

    # 去重
    terms = list(dict.fromkeys(terms))
    return terms if len(terms) > 0 else None


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)


def sae_decode_safe(sae, z):
    """
    兼容不同 sae_lens 版本：
    优先使用 sae.decode(z)，否则手动 z @ W_dec + b_dec
    z: [B, S, D_sae]
    return: [B, S, D_model]
    """
    if hasattr(sae, "decode"):
        return sae.decode(z)
    else:
        out = torch.einsum("bsd,dm->bsm", z, sae.W_dec)
        if hasattr(sae, "b_dec") and sae.b_dec is not None:
            out = out + sae.b_dec
        return out


def build_input_ids(tokenizer, prompt_text, device):
    """
    尽量稳妥地构造 chat 输入。
    """
    messages = [
        {
            "role": "user",
            "content": (
                "You are a chatbot that strictly follows commands. "
                "Do not output any additional information.\n\n"
                f"{prompt_text}"
            ),
        }
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(device)

    return input_ids


def find_subsequence_range(sequence, subsequence):
    """
    在 sequence 中查找 subsequence 的连续位置，返回 [start, end)。
    """
    seq_len = len(sequence)
    sub_len = len(subsequence)

    if sub_len == 0 or sub_len > seq_len:
        raise ValueError("Subsequence is empty or longer than sequence.")

    for start_idx in range(seq_len - sub_len + 1):
        if sequence[start_idx:start_idx + sub_len] == subsequence:
            return start_idx, start_idx + sub_len

    raise ValueError("Could not locate prompt_text token span inside chat template.")


def build_input_ids_with_prompt_span(tokenizer, prompt_text, device):
    """
    返回 chat template 的 input_ids，以及 prompt_text 在其中的 token span。
    """
    input_ids = build_input_ids(tokenizer, prompt_text, device)

    full_token_list = input_ids[0].detach().cpu().tolist()
    prompt_token_list = tokenizer.encode(prompt_text, add_special_tokens=False)
    prompt_start, prompt_end = find_subsequence_range(full_token_list, prompt_token_list)

    return input_ids, prompt_start, prompt_end


# =========================
# Model / SAE Load
# =========================
def load_model_and_sae():
    model = HookedTransformer.from_pretrained(
        model_name=MODEL_ID,
        device=DEVICE
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    sae, cfg_dict, _ = SAE.from_pretrained(
        release=RELEASE,
        sae_id=SAE_ID
    )
    sae = sae.to(DEVICE)

    hook_point = sae.cfg.hook_name

    print(f"[Info] model loaded: {MODEL_ID}")
    print(f"[Info] sae loaded: {RELEASE} / {SAE_ID}")
    print(f"[Info] hook_point: {hook_point}")
    print(f"[Info] device: {DEVICE}")

    return model, tokenizer, sae, hook_point


# =========================
# Feature Extraction
# =========================
def extract_word_sae_feature(model, sae, hook_point, text):
    """
    提取一个词/短语对应的 SAE 特征激活。
    返回: [1, seq, d_sae]
    """
    with torch.no_grad():
        _, cache = model.run_with_cache(text, prepend_bos=True)
        acts = sae.encode(cache[hook_point])
    return acts


def build_emotion_feature_container(model, sae, hook_point, top10_en):

    sae_feature_container = {}

    for idx, key in enumerate(top10_en.keys()):
        print(f"[Build] Processing emotion: {key}")

        words_to_process = [key] + top10_en[key]
        value_sae_save = []

        for word in words_to_process:
            value_sae_acts = extract_word_sae_feature(model, sae, hook_point, word)
            value_sae_acts = to_numpy(value_sae_acts)[0][1:]   
            features = np.sum(value_sae_acts, axis=0)        
            value_sae_save.append(features)

        value_sae_save = np.array(value_sae_save)
        print(f"[Build] {key} shape: {value_sae_save.shape}")

        sae_feature_container[idx] = value_sae_save

    return sae_feature_container


def clean_space(sae_feature_container):
    """
    去掉多个目标情绪的公共激活位点。
    """
    target_emo_ids = [4, 12, 16, 19, 23, 26]

    clean_sae_feature_container = copy.deepcopy(sae_feature_container)
    active_sets = []

    for emo_id in target_emo_ids:
        data = clean_sae_feature_container[emo_id]

        if isinstance(data, torch.Tensor):
            active_mask = (data > 0).any(dim=0)
            active_indices = set(torch.where(active_mask)[0].cpu().numpy())
        else:
            active_mask = (data > 0).any(axis=0)
            active_indices = set(np.where(active_mask)[0])

        active_sets.append(active_indices)
        print(f"[Clean] Emotion {emo_id} strictly activates {len(active_indices)} features.")

    common_features = set.intersection(*active_sets)
    common_features_list = list(common_features)

    print(f"[Clean] Common active features across selected emotions: {len(common_features_list)}")

    if len(common_features_list) > 0:
        for emo_id in target_emo_ids:
            clean_sae_feature_container[emo_id][:, common_features_list] = 0.0

    return clean_sae_feature_container

def get_steering_vectors_all(sae_feature_container, num):

    emo_feature_space = sae_feature_container[num]

    active_indices = np.unique(np.where(emo_feature_space != 0)[1])
    n_active_features = len(active_indices)

    print(n_active_features)

    return active_indices


def get_steering_vectors_plan_b(sae_feature_container, num, n_components=8, top_n_features=15):

    emo_feature_space = sae_feature_container[num]

    active_indices = np.unique(np.where(emo_feature_space != 0)[1])
    n_active_features = len(active_indices)

    if n_active_features == 0:
        print("[Warn] No active features found in this subspace.")
        return np.array([], dtype=np.int64)

    emo_feature_space_without_zeros = emo_feature_space[:, active_indices]

    actual_n_components = min(n_components, n_active_features)

    nmf = NMF(
        n_components=actual_n_components,
        init="random",
        random_state=42,
        max_iter=5000
    )

    _ = nmf.fit_transform(emo_feature_space_without_zeros)
    H = nmf.components_

    global_feature_importance = np.sum(H, axis=0)

    actual_top_n = min(top_n_features, n_active_features)
    best_local_indices = np.argsort(global_feature_importance)[::-1][:actual_top_n]
    new_steering_index = active_indices[best_local_indices]

    return new_steering_index.astype(np.int64)


def get_words_near_median(df, k):
    df = df.copy()
    median_value = df["V.Mean.Sum"].median()
    df["diff_from_median"] = abs(df["V.Mean.Sum"] - median_value)
    top_k_words = df.nsmallest(k, "diff_from_median")
    return top_k_words["Word"].tolist()


# =========================
# Steering
# =========================
def generate_results_steering(model, tokenizer, sae, hook_point, prompt_text, coeff, ster_index):

    pass



# =========================
# Erasure 
# =========================
def make_erasure_hook(sae, ster_index, prompt_start, prompt_end, alpha=1.0, edit_mode="delta", intervene_on="generation"):

    ster_index = torch.as_tensor(ster_index, device=DEVICE, dtype=torch.long)

    def erasure_hook(resid, hook):
        resid_new = resid.clone()
        prompt_len = prompt_end - prompt_start

        if intervene_on == "generation":
            if resid.shape[1] != 1:
                return resid
            pos_slice = slice(-1, None)

        elif intervene_on == "last_prompt":
            if resid.shape[1] == 1:
                return resid
            pos_slice = slice(prompt_end - 1, prompt_end)

        elif intervene_on == "prompt_tail":
            if resid.shape[1] == 1:
                return resid
            start_idx = prompt_start + max(0, int(prompt_len * 0.3))
            pos_slice = slice(start_idx, prompt_end)

        elif intervene_on == "all":
            if resid.shape[1] == 1:
                return resid
            pos_slice = slice(prompt_start, prompt_end)

        else:
            raise ValueError(f"Unknown intervene_on: {intervene_on}")

        x = resid[:, pos_slice, :]   # [B, S, d_model]

        z = sae.encode(x)            # [B, S, d_sae]
        z_edit = z.clone()

        # 核心：把目标情绪相关 feature 置零
        z_edit[..., ster_index] = 0.0

        if edit_mode == "replace":
            # h' = Dec(z')
            x_edit = sae_decode_safe(sae, z_edit)
            resid_new[:, pos_slice, :] = x_edit

        elif edit_mode == "delta":
            # h' = h + alpha * (Dec(z') - Dec(z))
            x_recon = sae_decode_safe(sae, z)
            x_edit = sae_decode_safe(sae, z_edit)
            delta = x_edit - x_recon
            resid_new[:, pos_slice, :] = x + alpha * delta

        elif edit_mode == "subtract":
            # h' = h - alpha * sum_f z_f d_f
            z_sel = z[..., ster_index]              # [B, S, K]
            w_sel = sae.W_dec[ster_index]           # [K, d_model]
            contribution = torch.einsum("bsk,km->bsm", z_sel, w_sel)
            resid_new[:, pos_slice, :] = x - alpha * contribution

        else:
            raise ValueError(f"Unknown edit_mode: {edit_mode}")

        return resid_new

    return erasure_hook


def generate_results_erasure(model, tokenizer, sae, hook_point, prompt_text, ster_index, alpha=1.0,
                             edit_mode="delta", intervene_on="generation"):
    input_ids, prompt_start, prompt_end = build_input_ids_with_prompt_span(tokenizer, prompt_text, DEVICE)
    terminators = get_terminators(tokenizer)

    model.reset_hooks()

    erasure_hook = make_erasure_hook(
        sae=sae,
        ster_index=ster_index,
        prompt_start=prompt_start,
        prompt_end=prompt_end,
        alpha=alpha,
        edit_mode=edit_mode,
        intervene_on=intervene_on
    )

    editing_hooks = [(hook_point, erasure_hook)]

    with model.hooks(fwd_hooks=editing_hooks):
        torch.manual_seed(SEED)
        results = model.generate(
            input_ids,
            max_new_tokens=250,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.1,
            top_p=0.1
        )

    generated_ids = results[0][input_ids.shape[1]:]
    res_str = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return res_str


# =========================
# Batch Generation
# =========================
def generate_sentence_with_index_steering(model, tokenizer, sae, hook_point, words, coeff, emo_id, ster_index):
    sentence_dict = {}

    for word in words:
        prompt_text = (
            f'Generate a coherent and realistic sentence that naturally incorporates the word "{word}". '
            f'Ensure the sentence uses the word correctly (grammatically and contextually) '
            f'and aligns with common sense, logic, and real-world knowledge.'
        )

        extracted_sentence = generate_results_steering(
            model=model,
            tokenizer=tokenizer,
            sae=sae,
            hook_point=hook_point,
            prompt_text=prompt_text,
            coeff=coeff,
            ster_index=ster_index
        )

        cleaned_sentence = extracted_sentence.replace("*", "").strip()
        sentence_dict[word] = cleaned_sentence

    return sentence_dict


def generate_sentence_with_index_erasure(model, tokenizer, sae, hook_point, words, emo_id, ster_index,
                                         alpha=1.0, edit_mode="delta", intervene_on="generation"):
    sentence_dict = {}

    for word in words:
        prompt_text = (
            f'Generate a coherent and realistic sentence that naturally incorporates the word "{word}". '
            f'Ensure the sentence uses the word correctly (grammatically and contextually) '
            f'and aligns with common sense, logic, and real-world knowledge. The sentence should convey a clear emotion of disgust.'
        )

        extracted_sentence = generate_results_erasure(
            model=model,
            tokenizer=tokenizer,
            sae=sae,
            hook_point=hook_point,
            prompt_text=prompt_text,
            ster_index=ster_index,
            alpha=alpha,
            edit_mode=edit_mode,
            intervene_on=intervene_on
        )

        cleaned_sentence = extracted_sentence.replace("*", "").strip()
        sentence_dict[word] = cleaned_sentence

    return sentence_dict


# =========================
# Optional: Single Feature Scan
# =========================
def single_feature_erasure_test(model, tokenizer, sae, hook_point, prompt_text, feature_id,
                                alpha=1.0, edit_mode="delta", intervene_on="generation"):

    return generate_results_erasure(
        model=model,
        tokenizer=tokenizer,
        sae=sae,
        hook_point=hook_point,
        prompt_text=prompt_text,
        ster_index=[feature_id],
        alpha=alpha,
        edit_mode=edit_mode,
        intervene_on=intervene_on
    )


# =========================
# Emotion Lexicon
# =========================
TOP10_EN = {
    "admiration": ["admire", "admiralty", "adulation", "adoration", "admiral", "appreciation", "adored", "praise", "idolize", "infatuated"],
    "adoration": ["adoring", "adulation", "adored", "admiration", "adore", "admire", "exaltation", "worship", "cherish", "lovable"],
    "appreciation": ["appreciate", "applaud", "applause", "appreciated", "admiration", "admire", "gratitude", "adulation", "thankfulness", "thankful"],
    "amusement": ["amusement park", "amusing", "pastime", "recreation", "entertaining", "bemusement", "amazement", "entertained", "entertain", "theme park"],
    "anger": ["angry", "rage", "irate", "enraged", "indignation", "wrath", "anguish", "indignant", "angst", "ire"],
    "anxiety": ["anxious", "nervousness", "fear", "worries", "depression", "worry", "phobia", "apprehensive", "apprehension", "worried"],
    "awe": ["awe", "awe-inspiring", "amazement", "amaze", "awesome", "amazed", "wow", "admiration", "astonishment", "adorable"],
    "awkwardness": ["unease", "uh", "autism", "puberty", "tension", "dance", "comic", "handshake", "reject"],
    "boredom": ["bore", "bored", "boring", "tedium", "tedious", "monotony", "ennui", "dreariness", "monotonous", "boreholes"],
    "calm": ["calmness", "calming", "tranquil", "serene", "tranquility", "placid", "quiet", "peaceful", "soothing", "serenity"],
    "confusion": ["confused", "confuse", "confuzzlement", "befuddlement", "befuddled", "disorientation", "chaos", "disarray", "puzzlement", "perplexed"],
    "crave": ["craving", "craze", "Craven", "appetite", "fiend", "yearning", "consecrate", "decadent", "brevity", "hoard"],
    "disgust": ["disgustful", "disgusting", "dismay", "disdain", "disappointment", "disgrace", "disdainful", "disprove", "disbelief", "displeased"],
    "compassion": ["compassionate", "compel", "comprehension", "generosity", "empathetic", "sympathy", "empathy", "conscience", "altruism", "guilt"],
    "entrapment": ["confinement", "predicament", "lure", "snare", "slavery", "blackmail", "entropy"],
    "excitement": ["excite", "excitado", "exuberance", "exceed", "Exuberance", "thrill", "enthusiasm", "thrilled", "thrilling", "ecstatic"],
    "fear": ["fearful", "fearless", "scared", "afraid", "terrified", "frightened", "terrifying", "frightening", "frighten", "scare"],
    "horror": ["horrific", "horrified", "scary movie", "scary", "terrifying", "horrible", "slasher film", "horrid", "horrendous", "frightful"],
    "interest": ["interested", "interesting", "usury", "fascinating", "fascination", "profit", "in", "you", "inquire", "apathy"],
    "joy": ["joyful", "joyous", "delight", "rejoice", "sorrow", "ecstatic", "hooray", "jolly", "elated", "happiness"],
    "nostalgia": ["reminiscing", "reminiscence", "reminisce", "reminiscience", "sorrow", "homesickness", "sentimentality", "sentimental", "yearning", "reunion"],
    "relief": ["relief as a feeling", "relieve", "relieved", "respite", "relive", "surcease", "remission", "soothing", "ease", "succor"],
    "romance": ["romance", "romantic", "amorous", "intimacy", "seduction", "in love", "marriage", "lovers", "flirt", "sorrow"],
    "sadness": ["sadness", "sad", "sad face", "sorrow", "melancholy", "mournful", "tragic", "tragedy", "unhappiness", "morose"],
    "satisfaction": ["satisfied", "satisfy", "gratification", "sated", "contentment", "fulfillment", "satisfactory", "pleased", "delight", "disappointed"],
    "desire": ["desirable", "wanting", "want", "crave", "craving", "wants", "wishes", "yearning", "desperation", "wanted"],
    "surprise": ["surprised", "surprising", "shocked", "shocking", "shocker", "astonishment", "shock", "amazement", "astonish", "startle"]
}

if __name__ == "__main__":
    
    set_seed(SEED)

    model, tokenizer, sae, hook_point = load_model_and_sae()


    # 1) emotion feature container
    sae_feature_container = build_emotion_feature_container(
        model=model,
        sae=sae,
        hook_point=hook_point,
        top10_en=TOP10_EN
    )

    # 2) emo space

    ster_index = get_steering_vectors_all(
        sae_feature_container,
        num=EMO_ID,
    )

    print(f"[Main] ster_index ({len(ster_index)}): {ster_index}")

    # 4) 取词
    df = pd.read_csv(CSV_PATH)
    words = get_words_near_median(df, K_WORDS)

    adj_for_para_words = ["refer", "staff", "flatten", "fold", "format", "layer", "origin", "close", "physician", "use"]
    new_words = [word for word in words if word not in adj_for_para_words]
    print(f"[Main] number of evaluation words: {len(new_words)}")

    results_all = {}

    if RUN_MODE == "steering":
        for alpha in ALPHA_LIST:
            print(f"[Run] steering coeff = {alpha}")
            sentence_dict = generate_sentence_with_index_steering(
                model=model,
                tokenizer=tokenizer,
                sae=sae,
                hook_point=hook_point,
                words=new_words,
                coeff=alpha,
                emo_id=EMO_ID,
                ster_index=ster_index
            )
            results_all[str(alpha)] = sentence_dict

    elif RUN_MODE == "erasure":
        for alpha in ALPHA_LIST:
            print(f"[Run] erasure alpha = {alpha}, edit_mode = {EDIT_MODE}, intervene_on = {INTERVENE_ON}")
            sentence_dict = generate_sentence_with_index_erasure(
                model=model,
                tokenizer=tokenizer,
                sae=sae,
                hook_point=hook_point,
                words=new_words,
                emo_id=EMO_ID,
                ster_index=ster_index,
                alpha=alpha,
                edit_mode=EDIT_MODE,
                intervene_on=INTERVENE_ON
            )
            results_all[str(alpha)] = sentence_dict
    else:
        raise ValueError(f"Unknown RUN_MODE: {RUN_MODE}")


    # 3) 选目标情绪相关 feature

    # 5) 保存
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results_all, f, indent=4, ensure_ascii=False)

    print(f"[Done] saved to: {OUTPUT_JSON}")
