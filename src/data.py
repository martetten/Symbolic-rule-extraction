import os
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import random
from collections import defaultdict
from torch.utils.data import Dataset
from transformer_lens import HookedTransformer
from datasets import load_dataset
import yaml

# Определяем корень проекта
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Базовый путь к данным RuleTaker (problog версия)
BASE_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "rultaker" / "rule-reasoning-dataset-V2020.2.5.0" / "problog"


def load_model_and_tokenizer(model_size="410m", device="cuda", dtype=torch.float16):
    name_map = {
        "410m": "EleutherAI/pythia-410m-deduped",
        "1b": "EleutherAI/pythia-1b-deduped",
        "qwen1.5b_instruct": "Qwen/Qwen2.5-1.5B-Instruct",
        "qwen1.5b_base": "Qwen/Qwen2.5-1.5B",
        "gpt2-large": "gpt2-large"
    }
    model_name = name_map[model_size]
    model = HookedTransformer.from_pretrained(
        model_name,
        dtype=dtype,
        device=device,
        fold_ln=False,
        center_writing_weights=False
    )

    tokenizer = model.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def load_rultaker(split="train", variant="depth-2", max_samples=None):
    """
    Загружает датасет RuleTaker (Problog версия) из локальной папки.
    split: "train", "dev", "test"
    variant: одна из папок внутри problog: depth-0, depth-1, depth-2, depth-3, depth-5, depth-3ext, birds-electricity, NatLang
    max_samples: ограничить количество примеров (для отладки)
    """
    folder = BASE_DATA_PATH / variant
    file_map = {
        "train": "train.jsonl",
        "dev": "dev.jsonl",
        "test": "test.jsonl"
    }
    file_path = folder / file_map[split]
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл {file_path} не найден. Проверьте путь и variant.")
    ds = load_dataset("json", data_files={split: str(file_path)}, split=split)
    if max_samples:
        ds = ds.select(range(max_samples))
    return ds

def prepare_example(example):
    """
    Приводит пример RuleTaker к унифицированному формату.
    """
    eng = example["english"]
    theory = " ".join(eng["theory_statements"])
    assertion = eng["assertion_statement"]
    full_text = f"{theory} {assertion}"
    label = example["theory_assertion_instance"]["label"]

    return {
        "text": full_text,
        "label": label,
        "theory": theory,
        "assertion": assertion,
        "id": example["id"],
        "min_proof_depth": example["theory_assertion_instance"].get("min_proof_depth", -1)
    }

def get_answer_token_ids(tokenizer, pos_words=("True", "true"), neg_words=("False", "false")):
    # Возвращает ID токенов для классов True и False
    # Использует ids[-1], чтобы избежать захвата токенов пробела-префикса (например, 'GTrue')
    def find_token(words):
        for word in words:
            ids = tokenizer.encode(word, add_special_tokens=False)
            if ids:
                return ids[-1]
        raise ValueError(f"Не найдено токенов для {words}")
    return find_token(pos_words), find_token(neg_words)


def evaluate_model_capability(model, dataset, prompt_template, batch_size=32, device="cuda"):
    # Универсальная zero-shot оценка способности модели решать задачу бинарной классификации
    # prompt_template передаётся напрямую из ноутбука, автоподбор убран
    model.eval()
    pos_id, neg_id = get_answer_token_ids(model.tokenizer)

    # Формируем входные тексты по переданному шаблону
    texts = [prompt_template.format(text=ex["text"]) for ex in dataset]
    labels = np.array([int(ex["label"]) for ex in dataset])

    # Предвычисляем длины промптов без специальных токенов
    # to_tokens добавит BOS самостоятельно, поэтому исходная длина = индекс последнего токена промпта
    prompt_lengths = [len(model.tokenizer.encode(t, add_special_tokens=False)) for t in texts]

    correct = 0
    total = 0

    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Evaluating capability"):
            batch_texts = texts[i:i+batch_size]
            batch_labels = labels[i:i+batch_size]
            batch_lengths = prompt_lengths[i:i+batch_size]

            # Токенизация с паддингом до макс. длины в батче и добавлением BOS
            tokens = model.to_tokens(batch_texts, prepend_bos=True).to(device)
            logits = model(tokens)

            # Извлекаем логиты последнего токена каждого промпта
            for j, seq_len in enumerate(batch_lengths):
                last_token_idx = seq_len
                logit_pos = logits[j, last_token_idx, pos_id].item()
                logit_neg = logits[j, last_token_idx, neg_id].item()

                pred = 1 if logit_pos > logit_neg else 0
                if pred == batch_labels[j]:
                    correct += 1
            total += len(batch_texts)

    return correct / total if total > 0 else 0.0

# def load_config(config_name: str = "experiment.yaml") -> dict:
#     """Загружает YAML конфиг из папки configs/."""
#     config_path = PROJECT_ROOT / "configs" / config_name
#     if not config_path.exists():
#         raise FileNotFoundError(f"Конфиг не найден: {config_path}")
#     with open(config_path, "r", encoding="utf-8") as f:
#         return yaml.safe_load(f)

def estimate_max_length(texts, tokenizer, percentile=95, sample_size=200):
    """Оценивает длину токенов на подвыборке и возвращает значение перцентиля."""
    sample = texts[:min(sample_size, len(texts))]
    encodings = tokenizer(sample, add_special_tokens=True, padding=False, truncation=False)
    lengths = [len(ids) for ids in encodings['input_ids']]
    max_len = int(np.percentile(lengths, percentile))
    print(f"{percentile}-й перцентиль: {max_len} токенов (при макс длине {max(lengths)})")
    return max_len

# def tokenize_dataset(dataset, tokenizer, max_length=128, text_key="text"):
#     """
#     Токенизирует датасет для подачи в модель.
#     """
#     def tokenize_fn(ex):
#         return tokenizer(
#             ex[text_key],
#             max_length=max_length,
#             padding="max_length",
#             truncation=True,
#             return_tensors="pt"
#         )
#     return dataset.map(tokenize_fn, batched=True)

class RuleTakerDataset(Dataset):
    """
    PyTorch Dataset для RuleTaker с предварительной токенизацией.
    Использует prepare_example для унификации.
    """
    def __init__(self, dataset, tokenizer, max_length=128, text_key="assertion_statement", label_key="label"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Применяем prepare_example к каждому элементу (если ещё не применено)
        if "text" not in dataset.column_names:
            dataset = dataset.map(prepare_example, remove_columns=dataset.column_names)
        self.dataset = dataset
        self.text_key = text_key
        self.label_key = label_key
        # Предварительная токенизация
        self._tokenize()

    def _tokenize(self):
        # Преобразуем Column в обычный список строк
        texts = list(self.dataset["text"])
        encodings = self.tokenizer(
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "label": self.dataset[idx]["label"],
            "text": self.dataset[idx]["text"],      # для аннотации
            "theory": self.dataset[idx]["theory"],  # для аннотации
            "id": self.dataset[idx]["id"]
        }

# def create_dataloader(dataset, tokenizer, batch_size=16, max_length=128, shuffle=True, num_workers=0):
#     """Создаёт DataLoader из сырого датасета RuleTaker."""
#     pytorch_dataset = RuleTakerDataset(dataset, tokenizer, max_length)
#     return DataLoader(pytorch_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def get_available_hook_names(model, max_layers=10):
    """Возвращает список имён хуков для первых max_layers слоёв (для отладки)."""
    hook_names = []
    for layer in range(min(max_layers, model.cfg.n_layers)):
        hook_names.extend([
            f"blocks.{layer}.hook_resid_pre",
            f"blocks.{layer}.hook_resid_post",
            f"blocks.{layer}.attn.hook_result",
            f"blocks.{layer}.mlp.hook_post",
        ])
    return hook_names

def create_minimal_pairs(variant="depth-1", n_pairs=30):
    """Генерирует пары (посылка + верно) vs (посылка + ложно) из датасета."""
    folder = BASE_DATA_PATH / variant
    file_path = folder / "train.jsonl"
    ds = load_dataset("json", data_files=str(file_path), split="train")

    by_premise = defaultdict(list)
    for ex in ds:
        premise = " ".join(ex["english"]["theory_statements"])
        assertion = ex["english"]["assertion_statement"]
        label = ex["theory_assertion_instance"]["label"]
        by_premise[premise].append({"assertion": assertion, "label": label})

    pairs = []
    for premise, assertions in by_premise.items():
        true_conds = [a["assertion"] for a in assertions if a["label"]]
        false_conds = [a["assertion"] for a in assertions if not a["label"]]
        if true_conds and false_conds:
            pairs.append({
                "premise": premise,
                "conclusion_true": random.choice(true_conds),
                "conclusion_false": random.choice(false_conds)
            })
        if len(pairs) >= n_pairs:
            break
    return pairs

def compute_differential_stats(model, tokenizer, pairs, layer_idx,
                               hook_name="resid_post", pooling="last",
                               device="cuda", max_length=256):
    """
    Разностный анализ на загруженной модели.
    Выполняет только один проход по парам.
    """
    model.eval()
    hook_path = f"blocks.{layer_idx}.hook_{hook_name}"
    diffs = []

    with torch.no_grad():
        for pair in pairs:
            txt_t = f"{pair['premise']} {pair['conclusion_true']}"
            txt_f = f"{pair['premise']} {pair['conclusion_false']}"

            tok_t = tokenizer(txt_t, return_tensors="pt", padding="max_length",
                              truncation=True, max_length=max_length).to(device)
            tok_f = tokenizer(txt_f, return_tensors="pt", padding="max_length",
                              truncation=True, max_length=max_length).to(device)

            _, cache_t = model.run_with_cache(tok_t["input_ids"], names_filter=[hook_path])
            _, cache_f = model.run_with_cache(tok_f["input_ids"], names_filter=[hook_path])

            act_t = cache_t[hook_path]
            act_f = cache_f[hook_path]

            pooled_t = act_t[:, -1, :] if pooling == "last" else act_t.mean(dim=1)
            pooled_f = act_f[:, -1, :] if pooling == "last" else act_f.mean(dim=1)

            diffs.append((pooled_t - pooled_f).float().cpu().numpy()[0])

    diffs = np.array(diffs)
    return {
        "mean_abs_diff": float(np.mean(np.abs(diffs))),
        "std_diff": float(np.std(np.abs(diffs))),
        "neuron_sensitivity": np.mean(np.abs(diffs), axis=0)
    }


def analyze_neuron_activations(model, tokenizer, layer_idx, neuron_idx,
                               texts, top_k_examples=10, top_k_tokens=30, device="cuda"):
    """
    Возвращает топ текстов и топ токенов, максимально активирующих заданный нейрон.
    За один прямой проход собирает оба набора данных.
    """
    model.eval()
    hook_path = f"blocks.{layer_idx}.hook_resid_post"

    text_activations = []
    token_acts = defaultdict(list)

    with torch.no_grad():
        for text in texts:
            # Токенизация
            encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
            input_ids = encoded["input_ids"][0]

            # Прямой проход с хуком
            _, cache = model.run_with_cache(input_ids, names_filter=[hook_path])
            resid = cache[hook_path][0]  # [seq_len, d_model]

            # Активации нейрона по всем токенам последовательности
            neuron_acts = resid[:, neuron_idx].float().cpu().numpy()

            # 1. Для текстов: берём активацию последнего токена
            text_activations.append((text, neuron_acts[-1]))

            # 2. Для токенов: мапим каждую активацию на соответствующий токен
            for tok_id, act in zip(input_ids.cpu().numpy(), neuron_acts):
                token = tokenizer.decode([tok_id])
                if token.strip() and not token.startswith("<"):
                    token_acts[token].append(act)

    # Сортировка и отбор топ-K текстов
    text_activations.sort(key=lambda x: x[1], reverse=True)
    top_texts = text_activations[:top_k_examples]

    # Агрегация (среднее) и отбор топ-K токенов
    token_scores = {tok: np.mean(acts) for tok, acts in token_acts.items()}
    top_tokens = sorted(token_scores.items(), key=lambda x: x[1], reverse=True)[:top_k_tokens]

    return top_texts, top_tokens
