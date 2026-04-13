# src/dataset_cot.py
# Функции для преобразования датасета RuleTaker в формат CoT (Chain of Thought)

import numpy as np
import torch
from tqdm import tqdm

def get_cot_prompt_template(model_name="default"):
    """
    Возвращает шаблон промпта с явной цепочкой рассуждений.
    Формат совместим с base-моделями (без чат-тегов).
    """
    # Базовый шаблон: теория + утверждение + пошаговый вывод + ответ
    return """Theory: {theory}
Assertion: {assertion}
Reasoning: Let's think step by step. {reasoning_steps}
Answer: {label}"""


def generate_reasoning_steps(theory, assertion, label, depth="depth-0"):
    """
    Генерирует текстовую цепочку рассуждений для примера.
    Упрощено для depth-0, расширяемо для depth-1+.
    """
    if label:
        if depth == "depth-0":
            return "The assertion is directly stated in the theory."
        else:
            return "From the theory, we can derive that the assertion must be true."
    else:
        if depth == "depth-0":
            return "The assertion contradicts the given facts."
        else:
            return "The theory does not support the assertion; the conditions are not met."


def format_example_cot(ex, template=None, depth="depth-0"):
    """
    Преобразует один пример RuleTaker в формат CoT для language modeling.
    """
    if template is None:
        template = get_cot_prompt_template()

    label_text = "True" if ex["label"] else "False"
    reasoning = generate_reasoning_steps(ex["theory"], ex["assertion"], ex["label"], depth=depth)

    formatted_text = template.format(
        theory=ex["theory"],
        assertion=ex["assertion"],
        reasoning_steps=reasoning,
        label=label_text
    )

    return {
        "text": formatted_text,
        "label": ex["label"],
        "id": ex.get("id", None)
    }

def prepare_cot_dataset(dataset, template=None, depth="depth-0", remove_original_cols=True):
    """
    Применяет CoT-форматирование ко всему датасету.
    """
    def map_fn(ex):
        return format_example_cot(ex, template, depth=depth)

    if hasattr(dataset, "keys") and "train" in dataset:
        return dataset.map(map_fn, remove_columns=dataset.column_names if remove_original_cols else None)
    else:
        return dataset.map(map_fn, remove_columns=dataset.column_names if remove_original_cols else None)


def evaluate_cot_capability(model, dataset, prompt_template, batch_size=16, device="cuda"):
    """
    Оценивает точность дообученной модели на датасете с CoT-структурой.
    prompt_template должен заканчиваться на 'Answer: ' без метки.
    """
    model.eval()

    # Определение ID токенов ответа
    true_ids = model.tokenizer.encode("True", add_special_tokens=False)
    false_ids = model.tokenizer.encode("False", add_special_tokens=False)
    true_id = true_ids[-1] if true_ids else 0
    false_id = false_ids[-1] if false_ids else 0

    correct = 0
    total = 0

    with torch.no_grad():
        for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating CoT capability"):
            # Определение границ батча
            end_idx = min(i + batch_size, len(dataset))
            batch = dataset.select(range(i, end_idx))

            # Формирование подсказок и меток
            prompts = [prompt_template.format(theory=ex["theory"], assertion=ex["assertion"]) for ex in batch]
            labels = np.array([int(ex["label"]) for ex in batch])

            # Токенизация через TransformerLens
            tokens = model.to_tokens(prompts, prepend_bos=True).to(device)
            logits = model(tokens)  # [batch, seq_len, vocab_size]

            # Длины промптов без спецтокенов (to_tokens добавляет BOS в начало)
            prompt_lengths = [len(model.tokenizer.encode(p, add_special_tokens=False)) for p in prompts]

            for j, seq_len in enumerate(prompt_lengths):
                # Последний токен промпта находится по индексу seq_len
                last_token_logits = logits[j, seq_len, :]
                logit_true = last_token_logits[true_id].item()
                logit_false = last_token_logits[false_id].item()

                pred = 1 if logit_true > logit_false else 0
                if pred == labels[j]:
                    correct += 1
            total += len(batch)

    return correct / total if total > 0 else 0.0

def format_direct_pythia(example):
    """
    Преобразует пример RuleTaker в текст для обучения Pythia в прямом формате.
    Формат: "{theory} {assertion} The assertion is{ True/False}"
    """
    theory = example["theory"]
    assertion = example["assertion"]
    label = example["label"]
    answer = " True" if label else " False"
    text = f"{theory} {assertion} The assertion is{answer}"
    return {"text": text, "label": label}

def prepare_direct_pythia_dataset(dataset, remove_original_cols=True):
    """Применяет format_direct_pythia ко всему датасету."""
    return dataset.map(
        format_direct_pythia,
        remove_columns=dataset.column_names if remove_original_cols else None
    )

def evaluate_direct_pythia(model, dataset, prompt_template, batch_size=16, device="cuda"):
    """
    Оценивает точность модели, обученной в прямом формате Pythia.
    prompt_template: "{theory} {assertion} The assertion is"
    """
    model.eval()
    # Используем токены с пробелом!
    true_id = model.tokenizer.encode(" True", add_special_tokens=False)[-1]
    false_id = model.tokenizer.encode(" False", add_special_tokens=False)[-1]

    correct = 0
    total = 0

    with torch.no_grad():
        for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating direct Pythia"):
            batch = dataset.select(range(i, min(i+batch_size, len(dataset))))
            prompts = [prompt_template.format(theory=ex["theory"], assertion=ex["assertion"]) for ex in batch]
            labels = np.array([int(ex["label"]) for ex in batch])

            tokens = model.to_tokens(prompts, prepend_bos=True).to(device)
            logits = model(tokens)

            # Длины промптов (без BOS)
            prompt_lengths = [len(model.tokenizer.encode(p, add_special_tokens=False)) for p in prompts]

            for j, seq_len in enumerate(prompt_lengths):
                last_logits = logits[j, seq_len, :]  # позиция сразу после промпта
                pred = 1 if last_logits[true_id] > last_logits[false_id] else 0
                if pred == labels[j]:
                    correct += 1
            total += len(batch)

    return correct / total if total > 0 else 0.0