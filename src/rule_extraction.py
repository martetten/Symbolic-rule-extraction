import torch
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier, export_text
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
from IPython.display import HTML, display

def compute_sae_activations(hooked_model, sae, tokenizer, dataset, layer_idx,
                            max_samples=500, device="cuda", use_eval_prefix=True):
    # Собирает активации SAE на последнем токене промпта.
    # use_eval_prefix=True применяет тот же шаблон, что в 04_baseline и evaluate_cot_capability
    hook_name = f"blocks.{layer_idx}.hook_resid_post"
    activations = []
    labels = []
    limit = min(max_samples, len(dataset))

    # Шаблон зафиксирован внутри для гарантии идентичности с baseline
    eval_prefix = "{theory} {assertion} The assertion is"

    with torch.no_grad():
        for i in tqdm(range(limit), desc="Вычисление активаций SAE"):
            item = dataset[i]
            label = 1 if item["label"] else 0

            if use_eval_prefix and "theory" in item and "assertion" in item:
                text = eval_prefix.format(theory=item["theory"], assertion=item["assertion"])
            else:
                text = item.get("text", "")

            tokens = hooked_model.to_tokens([text], prepend_bos=True).to(device)
            prompt_len = len(tokenizer.encode(text, add_special_tokens=False))
            safe_idx = min(prompt_len, tokens.shape[1] - 1)

            _, cache = hooked_model.run_with_cache(tokens, names_filter=lambda n: n == hook_name)
            acts = cache[hook_name]

            last_vec = acts[0, safe_idx, :].unsqueeze(0).to(torch.float32)
            sparse, _ = sae(last_vec)

            activations.append(sparse.cpu().numpy().flatten())
            labels.append(label)

    return np.array(activations), np.array(labels)

def compute_sae_activations_batched(
    hooked_model,
    sae,
    tokenizer,
    dataset,
    layer_idx,
    max_samples=500,
    batch_size=32,
    device="cuda",
    use_eval_prefix=True
):
    """
    Собирает латентные активации SAE на последнем токене промпта (пакетная версия).
    """
    hook_name = f"blocks.{layer_idx}.hook_resid_post"
    eval_prefix = "{theory} {assertion} The assertion is"

    limit = min(max_samples, len(dataset))
    all_sparse = []
    all_labels = []

    # Готовим все промпты заранее (как в исходной функции)
    texts = []
    labels_list = []
    for i in range(limit):
        item = dataset[i]
        if use_eval_prefix and "theory" in item and "assertion" in item:
            text = eval_prefix.format(theory=item["theory"], assertion=item["assertion"])
        else:
            text = item.get("text", "")
        texts.append(text)
        labels_list.append(1 if item["label"] else 0)

    # Обработка батчами
    with torch.no_grad():
        for start in tqdm(range(0, limit, batch_size), desc="Вычисление активаций SAE (batch)"):
            batch_texts = texts[start:start+batch_size]
            batch_labels = labels_list[start:start+batch_size]

            # Токенизируем с паддингом (to_tokens сам добавит BOS)
            tokens = hooked_model.to_tokens(batch_texts, prepend_bos=True).to(device)
            # Длины промптов (без BOS) для индексации последнего токена
            prompt_lens = [len(tokenizer.encode(t, add_special_tokens=False)) for t in batch_texts]

            # Прямой проход с кэшированием только нужного слоя
            _, cache = hooked_model.run_with_cache(tokens, names_filter=lambda n: n == hook_name)
            resid = cache[hook_name]   # [batch, seq_len, d_model]

            # Извлекаем векторы последнего токена для каждого примера
            last_vecs = []
            for j, seq_len in enumerate(prompt_lens):
                safe_idx = min(seq_len, resid.shape[1] - 1)   # защита от выхода за границу
                last_vecs.append(resid[j, safe_idx, :])
            last_vecs = torch.stack(last_vecs).to(torch.float32)   # [batch, d_model]

            # Прогон через SAE
            sparse, _ = sae(last_vecs)   # [batch, d_sae]

            all_sparse.append(sparse.cpu().numpy())
            all_labels.extend(batch_labels)

            del cache, tokens, resid, last_vecs, sparse
            torch.cuda.empty_cache()

    activations = np.concatenate(all_sparse, axis=0)  # [N, d_sae]
    labels = np.array(all_labels)
    return activations, labels

def rank_features_by_logic(activations, labels, dataset_texts):
    logic_keywords = ["if", "then", "and", "or", "not", "implies", "all", "some", "no"]
    has_logic = np.array([
        any(kw in txt.lower() for kw in logic_keywords) for txt in dataset_texts[:len(labels)]
    ], dtype=float)

    n_features = activations.shape[1]
    corr_label = np.zeros(n_features)
    corr_logic = np.zeros(n_features)

    for j in range(n_features):
        feat_vals = activations[:, j]
        if np.std(feat_vals) > 1e-6:
            corr_label[j] = np.corrcoef(feat_vals, labels)[0, 1]
            corr_logic[j] = np.corrcoef(feat_vals, has_logic)[0, 1]

    combined = np.abs(corr_label) * np.abs(corr_logic)
    top_features = np.argsort(combined)[::-1]
    return top_features, corr_label, corr_logic, combined

def generate_threshold_rules(activations, labels, corr_label, top_features, top_k=10):
    rules = []
    for feat_idx in top_features[:top_k]:
        threshold = float(np.median(activations[:, feat_idx]) + np.std(activations[:, feat_idx]))
        is_pos_corr = corr_label[feat_idx] > 0

        preds = (activations[:, feat_idx] > threshold).astype(int)
        if not is_pos_corr:
            preds = 1 - preds

        accuracy = float(np.mean(preds == labels))
        coverage = float(np.mean(activations[:, feat_idx] > threshold))

        rules.append({
            "feature_id": int(feat_idx),
            "threshold": threshold,
            "corr_with_label": float(corr_label[feat_idx]),
            "predicted_class": int(1 if is_pos_corr else 0),
            "rule_accuracy": accuracy,
            "coverage": coverage
        })
    return rules

def calculate_rule_fidelity(activations, rules, labels, min_coverage=0.01, min_accuracy=0.6):
    valid_rules = [
        r for r in rules
        if r["coverage"] >= min_coverage and r["rule_accuracy"] >= min_accuracy
    ]

    if not valid_rules:
        return 0.0, []

    explained = np.zeros(len(labels), dtype=bool)
    used_rules = []

    for rule in valid_rules:
        feat_idx = rule["feature_id"]
        threshold = rule["threshold"]
        is_pos_corr = rule["corr_with_label"] > 0

        preds = (activations[:, feat_idx] > threshold).astype(int)
        if not is_pos_corr:
            preds = 1 - preds

        rule_explains = (preds == labels)
        explained |= rule_explains
        used_rules.append(rule)

    fidelity = float(np.mean(explained))
    return fidelity, used_rules

def save_rule_results(exp_id, model_name, variant, layer_idx, sae_config, rules, fidelity, output_dir):
    out_path = Path(output_dir) / f"results/rules/{exp_id}_rules.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result_data = {
        "experiment_id": exp_id,
        "model": model_name,
        "variant": variant,
        "layer": layer_idx,
        "sae_config": sae_config,
        "rule_fidelity": fidelity,
        "rules": rules
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    print(f"Результаты сохранены: {out_path}")
    return out_path

def extract_semantic_annotations(activations, dataset, tokenizer, feature_ids, top_k=10):
    annotations = []
    for fid in feature_ids:
        top_indices = np.argsort(activations[:, fid])[::-1][:top_k]
        top_texts = [dataset[i]["text"] for i in top_indices]
        top_activations = activations[top_indices, fid]

        token_counts = {}
        for txt in top_texts:
            toks = tokenizer.tokenize(txt)
            for t in toks:
                clean_t = t.lstrip("\u0120").lower()
                if clean_t and not clean_t.startswith("<"):
                    token_counts[clean_t] = token_counts.get(clean_t, 0) + 1

        sorted_tokens = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:20]
        annotations.append({
            "feature_id": fid,
            "top_texts": top_texts,
            "top_activations": top_activations,
            "top_tokens": sorted_tokens
        })
    return annotations

def analyze_decision_tree_structure(activations, labels, feature_ids, max_depth=3):
    X_tree = activations[:, feature_ids]
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    dt.fit(X_tree, labels)
    tree_text = export_text(dt, feature_names=[f"Feature_{fid}" for fid in feature_ids])
    return tree_text, dt

def integrated_gradients_latent(hooked_model, sae, tokenizer, text, layer_idx, latent_idx, steps=50, device="cuda"):
    hooked_model.eval()
    sae.eval()

    tokens = hooked_model.to_tokens([text], prepend_bos=True).to(device)
    with torch.no_grad():
        embeds = hooked_model.embed(tokens).to(torch.float32)

    hook_point = f"blocks.{layer_idx}.hook_resid_post"
    baseline_embeds = torch.zeros_like(embeds)
    integrated_grads = torch.zeros_like(embeds)

    for step in tqdm(range(steps), desc=f"IG для латента {latent_idx}"):
        alpha = step / (steps - 1)
        interp = baseline_embeds + alpha * (embeds - baseline_embeds)
        interp = interp.clone().detach().requires_grad_(True)

        captured_act = None

        def embed_hook(module, input, output):
            return interp

        def resid_hook(module, input, output):
            nonlocal captured_act
            captured_act = output[0, -1, :]
            return output

        h_embed = hooked_model.hook_embed.register_forward_hook(embed_hook)
        h_resid = hooked_model.blocks[layer_idx].hook_resid_post.register_forward_hook(resid_hook)

        _ = hooked_model(tokens)

        h_embed.remove()
        h_resid.remove()

        if captured_act is None:
            raise RuntimeError("Не удалось захватить активацию resid_post")

        last_vec = captured_act.unsqueeze(0).to(torch.float32)
        sparse, _ = sae(last_vec)
        target_latent = sparse[0, latent_idx]

        target_latent.backward()
        integrated_grads += interp.grad
        interp.grad = None

    integrated_grads = integrated_grads / steps
    attributions = integrated_grads * (embeds - baseline_embeds)

    token_attrib = attributions.to(torch.float32).squeeze(0).norm(dim=1).detach().cpu().numpy()
    str_tokens = [tokenizer.decode([t]) for t in tokens[0]]
    return str_tokens, token_attrib

def visualize_token_attributions(tokens, values, title, cmap_name="YlOrRd", is_signed=False):
    cmap = plt.get_cmap(cmap_name)

    if is_signed:
        max_abs = np.max(np.abs(values)) if np.max(np.abs(values)) > 0 else 1.0
        norm = mcolors.TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
    else:
        vmin, vmax = np.percentile(values, [0, 95])
        if vmax == vmin:
            vmax = vmin + 1e-6
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    html_parts = []
    for tok, val in zip(tokens, values):
        mapped = norm(val)
        r, g, b, _ = cmap(mapped)
        alpha = 0.4 + 0.6 * (abs(val / max_abs) if is_signed else mapped)
        alpha = max(0.35, min(0.95, alpha))

        eff_r = r * alpha + (1.0 - alpha)
        eff_g = g * alpha + (1.0 - alpha)
        eff_b = b * alpha + (1.0 - alpha)
        brightness = 0.299*eff_r + 0.587*eff_g + 0.114*eff_b
        txt_color = "white" if brightness < 0.5 else "black"

        bg_color = f"rgba({int(r*255)}, {int(g*255)}, {int(b*255)}, {alpha:.2f})"
        html_parts.append(
            f'<span style="background-color:{bg_color}; color:{txt_color}; '
            f'padding:2px 0px; margin:1px; border-radius:3px; font-family:monospace;">{tok}</span>'
        )

    html_output = (
        f"<div style='line-height:1.5; padding:5px; background:#fff; border:1px solid #eee; "
        f"border-radius:3px; margin-bottom:15px;'>"
        f"<div style='font-weight:bold; margin-bottom:5px; font-size:12px;'>{title}</div>"
        f"{' '.join(html_parts)}</div>"
    )
    display(HTML(html_output))

def display_attribution_table(tokens, values, title, precision=4):
    df = pd.DataFrame({"Token": tokens, "Attribution Score": values})
    styled = (df.style
              .format(precision=precision)
              .set_caption(f"Числовые значения: {title}")
              .set_table_styles([{
                  'selector': 'caption',
                  'props': 'font-weight: bold; font-size: 0.9em; margin-bottom: 5px;'
              }]))
    display(styled)

def compute_dt_fidelity(acts_test, acts_train, preds_llm_test, preds_llm_train, labels_test, importance, k=20, max_depth=3):
    """Основная метрика: точность единого дерева (сопоставимо с baseline)"""
    top_k = np.argsort(np.abs(importance))[::-1][:k]
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    dt.fit(acts_train[:, top_k], preds_llm_train)
    preds = dt.predict(acts_test[:, top_k])
    fid = float(np.mean(preds == preds_llm_test))
    acc = float(np.mean(preds == labels_test))
    return fid, acc, dt

def compute_best_single_rule(acts_test, acts_train, target_test, target_train, importance_array, top_k_search=200):
    """Вспомогательная: лучшее пороговое правило (возвращает 2 значения)"""
    best_train_fid = -1.0
    best_info = None

    # Кандидаты отбираются по внешней важности (градиент/SHAP)
    candidates = np.argsort(np.abs(importance_array))[::-1][:top_k_search]

    for fid in candidates:
        threshold = float(np.median(acts_train[:, fid]) + np.std(acts_train[:, fid]))

        # Явный расчёт корреляции на train для определения направления правила
        corr = np.corrcoef(acts_train[:, fid], target_train)[0, 1]
        if np.isnan(corr):
            corr = 0.0

        preds_train = (acts_train[:, fid] > threshold).astype(int)
        if corr <= 0:
            preds_train = 1 - preds_train

        # Отбор по fidelity на обучающей выборке
        train_fid = float(np.mean(preds_train == target_train))
        if train_fid > best_train_fid:
            best_train_fid = train_fid
            best_info = {
                "feature_id": int(fid),
                "threshold": threshold,
                "corr": corr,
                "train_fid": train_fid
            }

    if best_info is None:
        return 0.0, None

    # Оценка на тесте
    fid = best_info["feature_id"]
    threshold = best_info["threshold"]
    preds_test = (acts_test[:, fid] > threshold).astype(int)
    if best_info["corr"] <= 0:
        preds_test = 1 - preds_test

    test_fid = float(np.mean(preds_test == target_test))
    return test_fid, best_info

def compute_union_coverage(acts_test, acts_train, target_test, target_train, importance, k, min_cov=0.01, min_acc=0.6):
    """Вспомогательная: доля примеров, покрытых хотя бы одним правилом (OR)"""
    top_k = np.argsort(np.abs(importance))[::-1][:k]
    rules = []
    for fid in top_k:
        thr = float(np.median(acts_train[:, fid]) + np.std(acts_train[:, fid]))
        p_train = (acts_train[:, fid] > thr).astype(int)
        corr = np.corrcoef(acts_train[:, fid], target_train)[0, 1]
        if corr <= 0: p_train = 1 - p_train
        acc = float(np.mean(p_train == target_train))
        cov = float(np.mean(acts_train[:, fid] > thr))
        if cov >= min_cov and acc >= min_acc:
            rules.append((fid, thr, corr))

    explained = np.zeros(len(acts_test), dtype=bool)
    for fid, thr, corr in rules:
        p_test = (acts_test[:, fid] > thr).astype(int)
        if corr <= 0: p_test = 1 - p_test
        explained |= (p_test == target_test)
    return float(np.mean(explained)), len(rules)