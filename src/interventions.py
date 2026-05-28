import torch
import numpy as np
import pandas as pd
from tqdm import tqdm


# def create_ablation_hook(sae, latent_idx, device="cuda"):
#     """Создаёт хук для обнуления целевого латента в residual stream."""
#     def hook(tensor, hook):
#         tensor = tensor.to(torch.float32)
#         last_vec = tensor[:, -1, :].unsqueeze(1)  # [batch, 1, d_model]
#         sparse, _ = sae(last_vec)
#         sparse[:, :, latent_idx] = 0.0
#         # Декодирование без b_dec для изоляции вклада направлений W_dec
#         recon = sparse @ sae.W_dec
#         tensor[:, -1, :] = recon.squeeze(1)
#         return tensor
#     return hook


# def create_patching_hook(sae, latent_idx, source_act, device="cuda"):
#     """Создаёт хук для замены активации латента на значение из источника."""
#     def hook(tensor, hook):
#         tensor = tensor.to(torch.float32)
#         last_vec = tensor[:, -1, :].unsqueeze(1)
#         sparse, _ = sae(last_vec)
#         sparse[:, :, latent_idx] = source_act.to(sparse.device)
#         recon = sparse @ sae.W_dec
#         tensor[:, -1, :] = recon.squeeze(1)
#         return tensor
#     return hook

def create_ablation_hook(sae, latent_idx, device="cuda"):
    """Создаёт хук для обнуления целевого латента в residual stream, реконструкция с b_dec."""
    def hook(tensor, hook):
        tensor = tensor.to(torch.float32)
        last_vec = tensor[:, -1, :].unsqueeze(1)
        sparse, _ = sae(last_vec)
        sparse[:, :, latent_idx] = 0.0
        # ПОЛНАЯ реконструкция: включая b_dec
        recon = sparse @ sae.W_dec + sae.b_dec + sae.mean
        tensor[:, -1, :] = recon.squeeze(1)
        return tensor
    return hook


def create_patching_hook(sae, latent_idx, source_act, device="cuda"):
    """Создаёт хук для замены активации латента на значение из источника, реконструкция с b_dec."""
    def hook(tensor, hook):
        tensor = tensor.to(torch.float32)
        last_vec = tensor[:, -1, :].unsqueeze(1)
        sparse, _ = sae(last_vec)
        sparse[:, :, latent_idx] = source_act.to(sparse.device)
        recon = sparse @ sae.W_dec + sae.b_dec + sae.mean
        tensor[:, -1, :] = recon.squeeze(1)
        return tensor
    return hook

def run_ablation(hooked_model, sae, texts, llm_preds, labels_gt, prompt_lens,
                 latent_indices, hook_name, true_id, false_id, batch_size=32, device="cuda"):
    """Замеряет эффект абляции для набора латентов. Возвращает pd.DataFrame."""
    results = []
    for lat_idx in tqdm(latent_indices, desc="Ablation"):
        hook_fn = create_ablation_hook(sae, lat_idx, device)
        pred_base, pred_interv = [], []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                tokens = hooked_model.to_tokens(texts[i:i+batch_size], prepend_bos=True).to(device)
                logits_b = hooked_model(tokens)
                logits_a = hooked_model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook_fn)])

                for b in range(tokens.shape[0]):
                    seq_len = prompt_lens[i+b]
                    safe_idx = min(seq_len, logits_b.shape[1] - 1)
                    pred_base.append(int(logits_b[b, safe_idx, true_id] > logits_b[b, safe_idx, false_id]))
                    pred_interv.append(int(logits_a[b, safe_idx, true_id] > logits_a[b, safe_idx, false_id]))

        pred_base = np.array(pred_base)
        pred_interv = np.array(pred_interv)
        fid_before = float(np.mean(pred_base == llm_preds))
        fid_after = float(np.mean(pred_interv == llm_preds))
        acc_before = float(np.mean(pred_base == labels_gt))
        acc_after = float(np.mean(pred_interv == labels_gt))
        flip_rate = float(np.mean(pred_base != pred_interv))

        results.append({
            "latent_idx": int(lat_idx),
            "fidelity_before": round(fid_before, 4),
            "fidelity_after": round(fid_after, 4),
            "delta_fidelity": round(fid_before - fid_after, 4),
            "accuracy_before": round(acc_before, 4),
            "accuracy_after": round(acc_after, 4),
            "delta_accuracy": round(acc_before - acc_after, 4),
            "flip_rate": round(flip_rate, 4)
        })
    return pd.DataFrame(results)


def run_patching(hooked_model, sae, texts, labels, prompt_lens, latent_indices,
                 hook_name, true_id, false_id, n_pairs=30, device="cuda"):
    """Проверяет достаточность латентов (False -> True патчинг). Возвращает pd.DataFrame."""
    results = []
    true_idx = np.where(labels == 1)[0]
    false_idx = np.where(labels == 0)[0]

    for lat_idx in tqdm(latent_indices, desc="Patching"):
        flips = 0
        tested = 0
        for _ in range(n_pairs):
            if len(true_idx) == 0 or len(false_idx) == 0:
                break
            src_i = np.random.choice(true_idx)
            tgt_i = np.random.choice(false_idx)

            # Извлечение активации источника с явным приведением типов и устройства
            src_tokens = hooked_model.to_tokens([texts[src_i]], prepend_bos=True).to(device)
            with torch.no_grad():
                _, cache = hooked_model.run_with_cache(src_tokens, names_filter=lambda n: n == hook_name)
                safe_src_idx = min(prompt_lens[src_i], cache[hook_name].shape[1] - 1)
                src_act_vec = cache[hook_name][0, safe_src_idx, :].to(torch.float32).to(device)

                # Кодирование в SAE -> извлечение скалярного значения латента
                src_sparse, _ = sae(src_act_vec.unsqueeze(0))  # [1, d_sae]
                source_latent_val = src_sparse[0, lat_idx].unsqueeze(0)  # [1]

            # Базовое предсказание цели
            tgt_tokens = hooked_model.to_tokens([texts[tgt_i]], prepend_bos=True).to(device)
            with torch.no_grad():
                logits_b = hooked_model(tgt_tokens)
                s = min(prompt_lens[tgt_i], logits_b.shape[1]-1)
                pred_b = int(logits_b[0, s, true_id] > logits_b[0, s, false_id])

                # Патчинг
                hook = create_patching_hook(sae, lat_idx, source_latent_val, device)
                logits_a = hooked_model.run_with_hooks(tgt_tokens, fwd_hooks=[(hook_name, hook)])
                pred_a = int(logits_a[0, s, true_id] > logits_a[0, s, false_id])

            if pred_b != pred_a:
                flips += 1
            tested += 1

        results.append({
            "latent_idx": int(lat_idx),
            "tested_pairs": tested,
            "flips": flips,
            "flip_rate": round(flips / max(tested, 1), 4),
            "direction": "False->True (sufficiency)"
        })
    return pd.DataFrame(results)


def compute_logit_shifts(hooked_model, sae, texts, prompt_lens, latent_idx, hook_name, true_id, false_id, batch_size=64, device="cuda"):
    """Возвращает массивы разности логитов до и после абляции."""
    diffs_b, diffs_a = [], []
    hook = create_ablation_hook(sae, latent_idx, device)
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            tokens = hooked_model.to_tokens(texts[i:i+batch_size], prepend_bos=True).to(device)
            logits_b = hooked_model(tokens)
            logits_a = hooked_model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook)])
            for b in range(tokens.shape[0]):
                s = min(prompt_lens[i+b], logits_b.shape[1]-1)
                diffs_b.append((logits_b[b, s, true_id] - logits_b[b, s, false_id]).item())
                diffs_a.append((logits_a[b, s, true_id] - logits_a[b, s, false_id]).item())
    return np.array(diffs_b), np.array(diffs_a)


def get_flipped_texts(hooked_model, sae, texts, labels, prompt_lens, latent_idx, hook_name, true_id, false_id, max_count=5, device="cuda"):
    """Возвращает тексты и метаданные примеров, где предсказание изменилось."""
    flipped = []
    hook = create_ablation_hook(sae, latent_idx, device)
    with torch.no_grad():
        for i in range(0, len(texts), 16):
            tokens = hooked_model.to_tokens(texts[i:i+16], prepend_bos=True).to(device)
            logits_b = hooked_model(tokens)
            logits_a = hooked_model.run_with_hooks(tokens, fwd_hooks=[(hook_name, hook)])
            for b in range(tokens.shape[0]):
                s = min(prompt_lens[i+b], logits_b.shape[1]-1)
                pred_b = int(logits_b[b, s, true_id] > logits_b[b, s, false_id])
                pred_a = int(logits_a[b, s, true_id] > logits_a[b, s, false_id])
                if pred_b != pred_a:
                    flipped.append({
                        "idx": int(i+b),
                        "text": texts[i+b][:300],
                        "gt": int(labels[i+b]),
                        "pred_before": pred_b,
                        "pred_after": pred_a
                    })
                    if len(flipped) >= max_count:
                        return flipped
    return flipped