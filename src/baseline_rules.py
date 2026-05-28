import torch
import numpy as np
from tqdm import tqdm

def compute_baseline_activations_and_preds(hooked_model, tokenizer, dataset, layer_idx, max_samples=2000, device="cuda"):
    hook_name = f"blocks.{layer_idx}.hook_resid_post"
    activations = []
    labels = []
    model_preds = []

    limit = min(max_samples, len(dataset))
    true_ids = tokenizer.encode("True", add_special_tokens=False)
    false_ids = tokenizer.encode("False", add_special_tokens=False)
    true_id = true_ids[-1] if true_ids else 0
    false_id = false_ids[-1] if false_ids else 0

    with torch.no_grad():
        for i in tqdm(range(limit), desc="Сбор baseline активаций и preds"):
            item = dataset[i]
            text = item["text"]
            label = 1 if item["label"] else 0

            tokens = hooked_model.to_tokens([text], prepend_bos=True).to(device)
            logits, cache = hooked_model.run_with_cache(tokens, names_filter=lambda n: n == hook_name)
            acts = cache[hook_name]

            prompt_len = len(tokenizer.encode(text, add_special_tokens=False))
            last_logits = logits[0, prompt_len, :]
            model_pred = 1 if last_logits[true_id] > last_logits[false_id] else 0
            model_preds.append(model_pred)

            safe_idx = min(prompt_len, acts.shape[1] - 1)
            activations.append(acts[0, safe_idx, :].detach().cpu().float().numpy())
            labels.append(label)

    return np.array(activations), np.array(model_preds), np.array(labels)