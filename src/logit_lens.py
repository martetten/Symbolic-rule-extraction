import numpy as np
import torch
import json
from src.data import PROJECT_ROOT

def get_target_token_ids(tokenizer):
    """
    Возвращает списки ID токенов для классов True и False.
    Учитывает разные регистры и возможные субтокены.
    """
    true_cands = ["true", "True", "yes", "Yes"]
    false_cands = ["false", "False", "no", "No"]

    true_ids = set()
    false_ids = set()

    for word in true_cands:
        ids = tokenizer.encode(word, add_special_tokens=False)
        if len(ids) == 1:
            true_ids.add(ids[0])

    for word in false_cands:
        ids = tokenizer.encode(word, add_special_tokens=False)
        if len(ids) == 1:
            false_ids.add(ids[0])

    # Fallback: если токены не нашлись, используем первые вхождения
    if not true_ids:
        true_ids = {tokenizer.encode("true", add_special_tokens=False)[0]}
    if not false_ids:
        false_ids = {tokenizer.encode("false", add_special_tokens=False)[0]}

    return list(true_ids), list(false_ids)

def compute_logit_lens(model, dataloader, n_layers, TRUE_TOKEN_IDS, FALSE_TOKEN_IDS, device="cuda"):
    """
    Вычисляет точность предсказания меток через Logit Lens для каждого слоя.
    Возвращает: layer_accs, layer_prob_true, layer_prob_false
    """
    layer_correct_sum = np.zeros(n_layers, dtype=np.float64)
    layer_total_samples = np.zeros(n_layers, dtype=np.float64)
    layer_prob_true_sum = np.zeros(n_layers, dtype=np.float64)
    layer_prob_false_sum = np.zeros(n_layers, dtype=np.float64)
    layer_abs_diff_sum = np.zeros(n_layers, dtype=np.float64)
    layer_confidence_sum = np.zeros(n_layers, dtype=np.float64)
    layer_entropy_sum = np.zeros(n_layers, dtype=np.float64)
    layer_agreement_sum = np.zeros(n_layers, dtype=np.float64)

    hook_names = [f"blocks.{i}.hook_resid_post" for i in range(n_layers)]

    model.eval()
    n_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].cpu().numpy()
            batch_size = len(input_ids)

            # Один проход кеширует все resid_post (и только их)
            _, cache = model.run_with_cache(input_ids, names_filter=hook_names)

            final_resid = cache[hook_names[-1]]
            final_ln = model.ln_final(final_resid)
            final_logits = model.unembed(final_ln)[:, -1, :]
            final_probs = torch.softmax(final_logits, dim=-1)
            final_pred = (
                final_probs[:, TRUE_TOKEN_IDS].sum(dim=-1) >
                final_probs[:, FALSE_TOKEN_IDS].sum(dim=-1)
            ).int().cpu().numpy()

            for i in range(n_layers):
                # Получаем остаточный поток после i-го слоя
                resid = cache[f"blocks.{i}.hook_resid_post"]

                # Logit Lens: применяем финальную нормализацию к промежуточному остатку
                resid_ln = model.ln_final(resid)

                # Проецируем в пространство словаря
                logits = model.unembed(resid_ln)  # [batch, seq, vocab]
                last_logits = logits[:, -1, :]    # Берём последний токен

                # Вычисляем вероятности для целевых токенов
                probs = torch.softmax(last_logits.float(), dim=-1)

                # Агрегируем вероятности по всем вариантам написания True/False
                prob_true = probs[:, TRUE_TOKEN_IDS].sum(dim=-1).cpu().numpy()
                prob_false = probs[:, FALSE_TOKEN_IDS].sum(dim=-1).cpu().numpy()

                # Предсказание: True если P(true) > P(false)
                preds = (prob_true > prob_false).astype(int)

                # Margin (абсолютная разница вероятностей)
                abs_diff = np.abs(prob_true - prob_false)

                # Уверенность модели (максимум из двух вероятностей)
                confidence = np.maximum(prob_true, prob_false)

                # Нормализованная бинарная энтропия
                # Вероятности нормируются относительно пары {True, False}
                p_sum = prob_true + prob_false + 1e-9
                p_norm = prob_true / p_sum
                eps = 1e-12
                p_safe = np.clip(p_norm, eps, 1.0 - eps)
                with np.errstate(divide='ignore', invalid='ignore'):
                    entropy = -p_safe * np.log2(p_safe) - (1.0 - p_safe) * np.log2(1.0 - p_safe)

                # Согласованность с финальным выводом модели
                agreement = (preds == final_pred).astype(float)

                # Накопление метрик
                layer_correct_sum[i] += np.sum(preds == labels)
                layer_total_samples[i] += batch_size
                layer_prob_true_sum[i] += np.sum(prob_true)
                layer_prob_false_sum[i] += np.sum(prob_false)
                layer_abs_diff_sum[i] += np.sum(abs_diff)
                layer_confidence_sum[i] += np.sum(confidence)
                layer_entropy_sum[i] += np.sum(entropy)
                layer_agreement_sum[i] += np.sum(agreement)

            # Очистка кэша после каждой пачки
            n_batches += 1
            del cache

    # Нормализация по количеству примеров
    total = layer_total_samples[0] + 1e-9

    return {
        "accs": layer_correct_sum / total,
        "mean_prob_true": layer_prob_true_sum / total,
        "mean_prob_false": layer_prob_false_sum / total,
        "mean_abs_diff": layer_abs_diff_sum / total,
        "mean_confidence": layer_confidence_sum / total,
        "mean_entropy": layer_entropy_sum / total,
        "mean_agreement": layer_agreement_sum / total,
        "n_samples": int(total)
    }

def load_logit_lens_results(model_size, variant, results_dir=None):
    """
    Загружает результаты Logit Lens из JSON-файла.
    Возвращает словарь с results с ключами: metadata, summary, layers, accs, mean_prob_true / mean_prob_false, mean_abs_diff, mean_confidence, mean_entropy, mean_agreement, n_samples
    """

    if results_dir is None:
        results_dir = PROJECT_ROOT / "results" / "logit_lens"
    fname = f"logit_lens_{model_size}_{variant}.json"
    fpath = results_dir / fname
    if not fpath.exists():
        raise FileNotFoundError(f"Файл {fpath} не найден. Сначала запустите эксперимент.")
    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data