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

MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
RELEASE = "llama_scope_lxr_32x"
SAE_ID = "l7r_32x"
EEMO_CSV_PATH = "./EEMO.csv"

K_NEAR_MEDIAN = 410
COEFF_CONTAINER = [0, 5, 10, 15, 20]

ADJ_FOR_PARA_WORDS = ["refer", "staff", "flatten", "fold", "format", "layer", "origin", "close", "physician", "use"]

TOP10_EN = {
    "admiration": ["admire", "adulation", "admiralty", "admiral", "adore", "adored", "adoration", "inspiration", "affection", "appreciation"],
    "adoration": ["adoring", "adorable", "adulation", "adore", "exaltation", "adored", "worship", "lovable", "endearment", "jubilation"],
    "appreciation": ["appreciate", "appreciated", "recognition", "perception", "enjoyment", "magnanimous", "compliment", "gratitude", "insight", "adulation"],
    "amusement": ["amusement park", "amusing", "amaze", "alcohol", "hilarity", "bemusement", "happiness", "recreation", "leisure", "trend"],
    "anger": ["angry", "wrath", "rage", "rebellion", "repression", "retribution", "happiness", "indignant", "discomfort", "spout"],
    "anxiety": ["anxious", "neurotic", "neurosis", "depression", "disaster", "surgery", "discomfort", "nervous system", "spiders", "fees"],
    "awe": ["awe", "awe-inspiring", "amazement", "reverence", "wonderment", "amaze", "amazed", "feeling", "awful", "grandeur"],
    "awkwardness": ["clumsy", "gawky", "uncomfortable", "ungainly", "embarrassing", "discomfort", "social interaction", "klutz", "unease", "anxiety"],
    "boredom": ["bored", "boring", "bore", "boreholes", "dreariness", "buzzer", "dullness", "tiresome", "monotony", "waste"],
    "calm": ["calmness", "calming", "calamari", "placid", "serene", "feeling", "lofty", "serenity", "exhale", "safeguard"],
    "confusion": ["confused", "confuse", "confuzzlement", "disbelief", "disorientation", "distraction", "explanation", "convince", "disoriented", "turmoil"],
    "crave": ["craving", "craze", "treat", "crisps", "hunger", "sweets", "chocolate", "consecrate", "salty", "wants"],
    "disgust": ["disgustful", "disgusting", "disdain", "revulsion", "dislike", "disgrace", "disbelief", "repugnant", "disdainful", "displeased"],
    "compassion": ["compassionate", "altruism", "generosity", "passion", "kindness", "humanity", "tenderness", "closeness", "mercy", "adoption"],
    "entrapment": ["predicament", "confinement", "lure", "snare", "blackmail", "slavery", "entropy"],
    "excitement": ["excitado", "excite", "arousal", "happening", "suspense", "exceed", "discovery", "uncontrollable", "eagerness", "thrilled"],
    "fear": ["fearful", "fearless", "fright", "flying", "fate", "flee", "facing", "frightened", "fret", "fury"],
    "horror": ["horrific", "horrid", "horrible", "horrified", "horrendous", "spooky", "scary movie", "politics", "disaster", "freight"],
    "interest": ["interested", "group", "care", "involved", "policy", "preference", "special", "field", "concern", "consuming"],
    "joy": ["joyous", "joyful", "Joyce", "rejoice", "serenity", "plentiful", "glory", "jolly", "gladness", "carefree"],
    "nostalgia": ["reminiscing", "reminiscence", "reminisce", "reminiscience", "sentimental", "reminse", "sentimentality", "retrospect", "melodrama", "wistful"],
    "relief": ["relieved", "relieve", "relief as a feeling", "relaxed", "relax", "relaxation", "relive", "recovery", "feeling", "rescue"],
    "romance": ["romance", "romantic", "amorous", "sorrow", "carriage", "hearts", "ambiance", "rescue", "marriage", "chocolate"],
    "sadness": ["sadness", "sad", "sad face", "sorrow", "melancholy", "lethargy", "unhappiness", "mournful", "happiness", "despondent"],
    "satisfaction": ["satisfied", "satisfactory", "satisfy", "sated", "mastery", "succeed", "happiness", "gratification", "sufficient", "sigh"],
    "desire": ["desirable", "desperation", "desperate", "feeling", "arousal", "belong", "crave", "perfection", "conquest", "reality"],
    "surprise": ["surprising", "surprised", "scare", "supersede", "secrets", "stun", "treat", "suspicious", "arrest", "shove"],
}

# {emo_id: [n_components, top_n_features]}
EMO_HYPERPARAMS = {
    4:  [5, 2],
    12: [5, 2],
    16: [5, 5],
    19: [5, 3],
    23: [5, 2],
    26: [7, 3],
}


EMO_PROMPT_START_POSITIONS = {
    4:  0.6,  # anger
    12: 0.5,  # disgust
    16: 0.8,  # fear
    19: 0.7,  # joy
    23: 0.6,  # sadness
    26: 0.3,  # surprise
}

# ------------------ Core helpers ------------------
@torch.no_grad()
def extract_word_sae_feature(model, sae, hook_point, prompt_text: str):
    _, cache = model.run_with_cache(prompt_text, prepend_bos=True)
    return sae.encode(cache[hook_point])


def get_steering_vectors(sae_feature_container, num, n_components=8, top_n_features=15):
    emo_feature_space = sae_feature_container[num]  # (10, D)

    active_indices = np.unique(np.where(emo_feature_space != 0)[1])
    n_active_features = len(active_indices)
    if n_active_features == 0:
        print(f"Warning: No active features found in subspace {num}.")
        return np.array([], dtype=int)

    emo_feature_space_without_zeros = emo_feature_space[:, active_indices]  # (10, active)

    actual_n_components = min(n_components, n_active_features)
    nmf = NMF(
        n_components=actual_n_components,
        init="random",
        random_state=42,
        max_iter=5000
    )

    _ = nmf.fit_transform(emo_feature_space_without_zeros)
    H = nmf.components_  # (n_components, active_features)

    global_feature_importance = np.sum(H, axis=0)  # (active_features,)
    actual_top_n = min(top_n_features, n_active_features)

    best_local_indices = np.argsort(global_feature_importance)[::-1][:actual_top_n]
    return active_indices[best_local_indices]


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

    # Lazy-load model/sae inside this function
    print("Loading model/tokenizer/SAE...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = HookedTransformer.from_pretrained(model_name=MODEL_ID, device=device)

    sae, _, _ = SAE.from_pretrained(release=RELEASE, sae_id=SAE_ID)
    sae = sae.to(device)

    # ------------------ 按要求硬编码 ------------------
    hook_point = "blocks.7.hook_resid_post"
    print("hook_point =", hook_point)

    # ---------- Build SAE feature container ----------
    sae_feature_container = {}
    for idx, key in enumerate(TOP10_EN.keys()):
        print(f"Processing: {key} (idx={idx})")

        words_to_process = [key] + TOP10_EN[key]  # 11 words
        value_sae_save = []

        for word in words_to_process:
            value_sae_acts = extract_word_sae_feature(model, sae, hook_point, word)
            value_sae_acts = value_sae_acts.detach().to(torch.float).cpu().numpy()[0][1:]  # remove BOS
            features = np.sum(value_sae_acts, axis=0)
            value_sae_save.append(features)

        value_sae_save = np.array(value_sae_save)  # (11, D)
        sae_feature_container[idx] = value_sae_save

    new_sae_feature_container = clean_space(sae_feature_container)

    # ---------- Get steering index ----------
    n_components, top_n_features = EMO_HYPERPARAMS[emo_id]
    ster_index = get_steering_vectors(
        new_sae_feature_container,
        num=emo_id,
        n_components=n_components,
        top_n_features=top_n_features
    )
    print(f"emo_id={emo_id}, n_components={n_components}, top_n_features={top_n_features}")
    print(f"prompt_start_position={prompt_start_position}")
    print(f"ster_index (len={len(ster_index)}): {ster_index}")

    # ---------- Load words and generate ----------
    df = pd.read_csv(EEMO_CSV_PATH)
    words = get_words_near_median(df, K_NEAR_MEDIAN)
    new_words = [w for w in words if w not in ADJ_FOR_PARA_WORDS]

    def generate_results(prompt_word, coeff, if_steering=True):
        steering_vector = torch.zeros_like(sae.W_dec[0])
        for idx_ in ster_index:
            steering_vector += sae.W_dec[idx_]

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
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(device)

        terminators = [
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]

        model.reset_hooks()
        # ------------------ 按要求硬编码 ------------------
        editing_hooks = [("blocks.7.hook_resid_post", steering_hook)]

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
        return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

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
            extracted_sentence = extracted_sentence.replace("<|eot_id|>", "").strip() if extracted_sentence else ""
            sentence_dict[word] = extracted_sentence
        return sentence_dict

    coeff_sentence_dict = {}
    for coeff in COEFF_CONTAINER:
        print(f"\nGenerating for coeff={coeff} ...")
        coeff_sentence_dict[coeff] = generate_sentence_with_index(new_words, coeff)

    # ---------- Save ----------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(coeff_sentence_dict, f, indent=4)

    print(f"\nSaved to: {output_path}")


# ------------------ Example usage ------------------
if __name__ == "__main__":
    Emo_id = 23
    run_for_emo(
        emo_id=Emo_id,
        output_path=f"./sentence_save_llama/num_{Emo_id}_k_{K_NEAR_MEDIAN}_clean.json"
    )