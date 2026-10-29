import os
import json
import copy
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import NMF
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer
from sae_lens import SAE

# ------------------ Fixed configuration ------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda")

MODEL_ID = "google/gemma-2-9b-it"
RELEASE = "gemma-scope-9b-it-res"
SAE_ID = "layer_9/width_16k/average_l0_88"
EEMO_CSV_PATH = "/data/xiuwen/google/files/EEMO.csv"

K_NEAR_MEDIAN = 410
COEFF_CONTAINER = [0, 5, 10, 15, 20]

ADJ_FOR_PARA_WORDS = ["refer", "staff", "flatten", "fold", "format", "layer", "origin", "close", "physician", "use"]

# {emo_id: [n_components, top_n_features]}
EMO_HYPERPARAMS = {
    4:  [11, 40],
    12: [6, 35],
    16: [11, 39],
    19: [5, 31],
    23: [5, 40],
    26: [9, 36],
}

# Prompt steering covers [int(p * start_position):int(p)].
# Smaller values start earlier; 0.0 starts at 0 and 1.0 selects no positions.
EMO_PROMPT_START_POSITIONS = {
    4:  0.6,  # anger
    12: 0.5,  # disgust
    16: 0.5,  # fear
    19: 0.5,  # joy
    23: 0.5,  # sadness
    26: 0.6,  # surprise
}

TOP10_EN = {
    'admiration': ['admire', 'admiralty', 'adulation', 'adoration', 'admiral', 'appreciation', 'adored', 'praise', 'idolize', 'infatuated'],
    'adoration': ['adoring', 'adulation', 'adored', 'admiration', 'adore', 'admire', 'exaltation', 'worship', 'cherish', 'lovable'],
    'appreciation': ['appreciate', 'applaud', 'applause', 'appreciated', 'admiration', 'admire', 'gratitude', 'adulation', 'thankfulness', 'thankful'],
    'amusement': ['amusement park', 'amusing', 'pastime', 'recreation', 'entertaining', 'bemusement', 'amazement', 'entertained', 'entertain', 'theme park'],
    'anger': ['angry', 'rage', 'irate', 'enraged', 'indignation', 'wrath', 'anguish', 'indignant', 'angst', 'ire'],
    'anxiety': ['anxious', 'nervousness', 'fear', 'worries', 'depression', 'worry', 'phobia', 'apprehensive', 'apprehension', 'worried'],
    'awe': ['awe', 'awe-inspiring', 'amazement', 'amaze', 'awesome', 'amazed', 'wow', 'admiration', 'astonishment', 'adorable'],
    'awkwardness': ['unease', 'uh', 'autism', 'puberty', 'tension', 'dance', 'comic', 'handshake', 'reject'],
    'boredom': ['bore', 'bored', 'boring', 'tedium', 'tedious', 'monotony', 'ennui', 'dreariness', 'monotonous', 'boreholes'],
    'calm': ['calmness', 'calming', 'tranquil', 'serene', 'tranquility', 'placid', 'quiet', 'peaceful', 'soothing', 'serenity'],
    'confusion': ['confused', 'confuse', 'confuzzlement', 'befuddlement', 'befuddled', 'disorientation', 'chaos', 'disarray', 'puzzlement', 'perplexed'],
    'crave': ['craving', 'craze', 'Craven', 'appetite', 'fiend', 'yearning', 'consecrate', 'decadent', 'brevity', 'hoard'],
    'disgust': ['disgustful', 'disgusting', 'dismay', 'disdain', 'disappointment', 'disgrace', 'disdainful', 'disprove', 'disbelief', 'displeased'],
    'compassion': ['compassionate', 'compel', 'comprehension', 'generosity', 'empathetic', 'sympathy', 'empathy', 'conscience', 'altruism', 'guilt'],
    'entrapment': ['confinement', 'predicament', 'lure', 'snare', 'slavery', 'blackmail', 'entropy'],
    'excitement': ['excite', 'excitado', 'exuberance', 'exceed', 'Exuberance', 'thrill', 'enthusiasm', 'thrilled', 'thrilling', 'ecstatic'],
    'fear': ['fearful', 'fearless', 'scared', 'afraid', 'terrified', 'frightened', 'terrifying', 'frightening', 'frighten', 'scare'],
    'horror': ['horrific', 'horrified', 'scary movie', 'scary', 'terrifying', 'horrible', 'slasher film', 'horrid', 'horrendous', 'frightful'],
    'interest': ['interested', 'interesting', 'usury', 'fascinating', 'fascination', 'profit', 'in', 'you', 'inquire', 'apathy'],
    'joy': ['joyful', 'joyous', 'delight', 'rejoice', 'sorrow', 'ecstatic', 'hooray', 'jolly', 'elated', 'happiness'],
    'nostalgia': ['reminiscing', 'reminiscence', 'reminisce', 'reminiscience', 'sorrow', 'homesickness', 'sentimentality', 'sentimental', 'yearning', 'reunion'],
    'relief': ['relief as a feeling', 'relieve', 'relieved', 'respite', 'relive', 'surcease', 'remission', 'soothing', 'ease', 'succor'],
    'romance': ['romance', 'romantic', 'amorous', 'intimacy', 'seduction', 'in love', 'marriage', 'lovers', 'flirt', 'sorrow'],
    'sadness': ['sadness', 'sad', 'sad face', 'sorrow', 'melancholy', 'mournful', 'tragic', 'tragedy', 'unhappiness', 'morose'],
    'satisfaction': ['satisfied', 'satisfy', 'gratification', 'sated', 'contentment', 'fulfillment', 'satisfactory', 'pleased', 'delight', 'disappointed'],
    'desire': ['desirable', 'wanting', 'want', 'crave', 'craving', 'wants', 'wishes', 'yearning', 'desperation', 'wanted'],
    'surprise': ['surprised', 'surprising', 'shocked', 'shocking', 'shocker', 'astonishment', 'shock', 'amazement', 'astonish', 'startle'],
}

# ------------------ Core helpers ------------------
@torch.no_grad()
def extract_word_sae_feature(model, sae, hook_point, prompt_text: str):
    _, cache = model.run_with_cache(prompt_text, prepend_bos=True)
    return sae.encode(cache[hook_point])


def get_steering_vectors(sae_feature_container, num, n_components=8, top_n_features=15):

    emo_feature_space = sae_feature_container[num] 

    active_indices = np.unique(np.where(emo_feature_space != 0)[1])
    n_active_features = len(active_indices)
    if n_active_features == 0:
        print(f"Warning: No active features found in subspace {num}.")
        return np.array([], dtype=int)

    emo_feature_space_without_zeros = emo_feature_space[:, active_indices]  

    actual_n_components = min(n_components, n_active_features)
    nmf = NMF(
        n_components=actual_n_components,
        init="random",
        random_state=42,
        max_iter=5000
    )

    _ = nmf.fit_transform(emo_feature_space_without_zeros)
    H = nmf.components_  # (n_components, active_features)

    global_feature_importance = np.sum(H, axis=0)
    actual_top_n = min(top_n_features, n_active_features)

    best_local_indices = np.argsort(global_feature_importance)[::-1][:actual_top_n]
    return active_indices[best_local_indices]


def get_steering_vectors_by_emo_params(sae_feature_container, emo_id):
    if emo_id not in EMO_HYPERPARAMS:
        raise ValueError(f"emo_id={emo_id} not found in EMO_HYPERPARAMS. Keys={list(EMO_HYPERPARAMS.keys())}")
    n_components, top_n_features = EMO_HYPERPARAMS[emo_id]
    return get_steering_vectors(
        sae_feature_container,
        num=emo_id,
        n_components=n_components,
        top_n_features=top_n_features
    )


def get_words_near_median(df, k):
    median_value = df["V.Mean.Sum"].median()
    df = df.copy()
    df["diff_from_median"] = abs(df["V.Mean.Sum"] - median_value)
    return df.nsmallest(k, "diff_from_median")["Word"].tolist()


def clean_space(sae_feature_container):
    target_emo_ids = [4, 12, 16, 19, 23, 26]
    clean_sae_feature_container = copy.deepcopy(sae_feature_container)

    active_sets = []
    for emo_id in target_emo_ids:
        data = clean_sae_feature_container[emo_id]  # (10, D)

        if isinstance(data, torch.Tensor):
            active_mask = (data > 0).any(dim=0)
            active_indices = set(torch.where(active_mask)[0].cpu().numpy())
        else:
            active_mask = (data > 0).any(axis=0)
            active_indices = set(np.where(active_mask)[0])

        active_sets.append(active_indices)
        print(f"Emotion {emo_id} strictly activates {len(active_indices)} features.")

    common_features = set.intersection(*active_sets)
    common_features_list = list(common_features)
    print(f"\n🔥 Strict common activation: {len(common_features_list)} features active in all selected emotions.")

    if len(common_features_list) > 0:
        for emo_id in target_emo_ids:
            clean_sae_feature_container[emo_id][:, common_features_list] = 0.0

    return clean_sae_feature_container


def run_for_emo(emo_id: int, output_path: str):
    if emo_id not in EMO_HYPERPARAMS:
        raise ValueError(f"Unknown emo_id={emo_id}. Allowed: {sorted(EMO_HYPERPARAMS.keys())}")

    if emo_id not in EMO_PROMPT_START_POSITIONS:
        raise ValueError(f"emo_id={emo_id} not found in EMO_PROMPT_START_POSITIONS.")
    prompt_start_position = EMO_PROMPT_START_POSITIONS[emo_id]
    if not 0.0 <= prompt_start_position <= 1.0:
        raise ValueError(
            f"EMO_PROMPT_START_POSITIONS[{emo_id}] must be between 0.0 and 1.0, "
            f"got {prompt_start_position!r}."
        )

    # Lazy-load model/tokenizer/SAE
    print("Loading model/tokenizer/SAE...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = HookedTransformer.from_pretrained(model_name=MODEL_ID, device=device)

    sae, _, _ = SAE.from_pretrained(release=RELEASE, sae_id=SAE_ID)
    sae = sae.to(device)

    hook_point = "blocks.9.hook_resid_post"
    print("hook_point =", hook_point)

    sae_feature_container = {}
    for idx, key in enumerate(TOP10_EN.keys()):
        print(f"Processing: {key} (idx={idx})")

        words_to_process = [key] + TOP10_EN[key] 
        value_sae_save = []

        for word in words_to_process:
            value_sae_acts = extract_word_sae_feature(model, sae, hook_point, word)
            value_sae_acts = value_sae_acts.detach().to(torch.float).cpu().numpy()[0][1:]  # remove BOS
            features = np.sum(value_sae_acts, axis=0)
            value_sae_save.append(features)

        value_sae_save = np.array(value_sae_save) 
        sae_feature_container[idx] = value_sae_save

    new_sae_feature_container = clean_space(sae_feature_container)

    # ---------- Get steering index ----------
    ster_index = get_steering_vectors_by_emo_params(new_sae_feature_container, emo_id=emo_id)
    n_components, top_n_features = EMO_HYPERPARAMS[emo_id]
    print(f"emo_id={emo_id}, n_components={n_components}, top_n_features={top_n_features}")
    print(f"prompt_start_position={prompt_start_position}")
    print(f"ster_index (len={len(ster_index)}): {ster_index}")

    # ---------- Load words and generate ----------
    df = pd.read_csv(EEMO_CSV_PATH)
    words = get_words_near_median(df, K_NEAR_MEDIAN)
    new_words = [w for w in words if w not in ADJ_FOR_PARA_WORDS]

    def generate_results(prompt_word, coeff, if_steering=True):
        steering_vector = torch.zeros_like(sae.W_dec[0])
        for idx in ster_index:
            steering_vector += sae.W_dec[idx]

        raw_prompt_tokens = model.to_tokens(prompt_word, prepend_bos=True)
        position = raw_prompt_tokens.shape[1]
        p = position - 1

        def steering_hook(resid_pre, hook):
            if resid_pre.shape[1] == 1:
                return
            if if_steering:
                start_idx = int(p * prompt_start_position)
                end_idx = int(p)
                resid_pre[:, start_idx:end_idx, :] += coeff * steering_vector

        messages = [
            {"role": "system", "content": "You are a chatbot that strictly follows commands. Do not output any additional information."},
            {"role": "user", "content": prompt_word},
        ]

        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(device)

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        model.reset_hooks()
        editing_hooks = [("blocks.9.hook_resid_post", steering_hook)]  # 也按要求保持硬编码

        with model.hooks(fwd_hooks=editing_hooks):
            torch.manual_seed(42)
            results = model.generate(
                input_ids,
                max_new_tokens=250,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.1,
                top_p=0.1,
            )

        generated_ids = results[0][input_ids.shape[1]:]
        res_str = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return res_str

    def generate_sentence_with_index(words_list, coeff):
        sentence_dict = {}
        for word in words_list:
            prompt_word = (
                f'Generate a coherent and realistic sentence that naturally incorporates the word "{word}". '
                f'Ensure the sentence uses the word correctly (grammatically and contextually) and aligns with common sense, '
                f'logic, and real-world knowledge.'
            )

            extracted_sentence = generate_results(
                prompt_word=prompt_word,
                coeff=coeff,
                if_steering=True
            )

            extracted_sentence = extracted_sentence.replace("*", "").strip() if extracted_sentence else ""
            sentence_dict[word] = extracted_sentence
        return sentence_dict

    coeff_sentence_dict = {}
    for coeff in COEFF_CONTAINER:
        print(f"\nGenerating for coeff={coeff} ...")
        coeff_sentence_dict[coeff] = generate_sentence_with_index(new_words, coeff)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(coeff_sentence_dict, f, indent=4)

    print(f"\nSaved to: {output_path}")


# ------------------ Example usage ------------------
if __name__ == "__main__":
    Emo_id = 19
    run_for_emo(
        emo_id=Emo_id,
        output_path=f"./sentence_save_gemma/num_{Emo_id}_k_{K_NEAR_MEDIAN}_clean.json"
    )