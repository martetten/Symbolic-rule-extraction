import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import json
from src.data import PROJECT_ROOT

def collect_activations(
    model: HookedTransformer,
    dataloader: DataLoader,
    layer_idx,
    hook_name = "resid_post",
    pooling = "last",
    device = "cuda",
    save_path=None
):
    """
    Собирает активации указанного слоя с оптимизацией памяти.
    """
    hook_path = f"blocks.{layer_idx}.hook_{hook_name}"
    all_activations = []
    all_labels = []

    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"]

            # names_filter кэширует только нужный хук, экономя 80-90% VRAM
            _, cache = model.run_with_cache(input_ids, names_filter=[hook_path])
            layer_activations = cache[hook_path]  # [batch, seq, d_model]

            if pooling == "last":
                pooled = layer_activations[:, -1, :].float().cpu().numpy()
            elif pooling == "mean":
                pooled = layer_activations.mean(dim=1).float().cpu().numpy()
            else:
                raise ValueError("pooling должен быть 'last' или 'mean'")

            all_activations.append(pooled)
            all_labels.extend(labels)

            del cache
            torch.cuda.empty_cache()

    acts_np = np.vstack(all_activations)

    # Сохранение только если явно указан путь
    if save_path is not None:
        save_path = Path(save_path).resolve()
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch.from_numpy(acts_np), save_path.with_suffix(".pt"))
        np.save(save_path.with_suffix(".labels.npy"), np.array(all_labels))
        print(f"Активации сохранены: {save_path.with_suffix('.pt')}")

    return acts_np, np.array(all_labels)


def compute_matrix_metrics(acts):
    """Вычисляет дополнительные метрики матрицы активаций."""
    try:
        # SVD-разложение для всех метрик
        U, S, Vt = np.linalg.svd(acts, full_matrices=False)

        # Число обусловленности через сингулярные значения (быстрее чем np.linalg.cond)
        min_s = np.min(S)
        cond_num = float(np.max(S) / min_s) if min_s > 1e-10 else float('inf')

        # Эффективный ранг (энтропия сингулярных значений)
        # Сдвиг для численной стабильности softmax
        S_shifted = S - np.max(S)
        exp_S = np.exp(S_shifted)
        softmax_S = exp_S / np.sum(exp_S)
        eff_rank = float(np.exp(-np.sum(softmax_S * np.log(softmax_S + 1e-12))))

        return {
            "condition_number": cond_num,
            "effective_rank": eff_rank,
            "max_singular_value": float(np.max(S)),
            "min_singular_value": float(min_s)
        }
    except Exception:
        return {
            "condition_number": float('inf'),
            "effective_rank": 0.0,
            "max_singular_value": 0.0,
            "min_singular_value": 0.0
        }


def run_probing_for_layer(model, train_loader, dev_loader, layer_idx, hook_name="resid_post", pooling="last"):
    """
    Собирает активации и обучает probing-классификатор для одного слоя.
    Возвращает: train_acc, dev_acc, train_stats, dev_stats
    """

    # Сбор активаций
    train_acts, train_lbls = collect_activations(model, train_loader, layer_idx, hook_name, pooling)
    dev_acts, dev_lbls = collect_activations(model, dev_loader, layer_idx, hook_name, pooling)

    # Матричные метрики
    train_acts = train_acts.astype(np.float32)  # предварительная конвертация в np.float32 для работы с np.linalg в compute_matrix_metrics
    dev_acts = dev_acts.astype(np.float32)
    train_matrix_metrics = compute_matrix_metrics(train_acts)
    dev_matrix_metrics = compute_matrix_metrics(dev_acts)

    # Обучение классификатора
    clf = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    clf.fit(train_acts, train_lbls)

    # Метрики качества
    train_acc = accuracy_score(train_lbls, clf.predict(train_acts))
    dev_acc = accuracy_score(dev_lbls, clf.predict(dev_acts))

    # Статистика активаций
    train_stats = {
        "mean_abs": float(np.mean(np.abs(train_acts))),
        "std_abs": float(np.std(np.abs(train_acts))),
        "max_abs": float(np.max(np.abs(train_acts))),
        "min_abs": float(np.min(np.abs(train_acts))),
        "sparsity": float(np.mean(np.abs(train_acts) < 0.01)),  # % почти нулевых активаций
        "mean_by_neuron": float(np.mean(np.std(train_acts, axis=0)))  # вариативность по нейронам
    }

    dev_stats = {
        "mean_abs": float(np.mean(np.abs(dev_acts))),
        "std_abs": float(np.std(np.abs(dev_acts))),
        "max_abs": float(np.max(np.abs(dev_acts))),
        "min_abs": float(np.min(np.abs(dev_acts))),
        "sparsity": float(np.mean(np.abs(dev_acts) < 0.01)),
        "mean_by_neuron": float(np.mean(np.std(dev_acts, axis=0)))
    }

    return train_acc, dev_acc, {**train_stats, **train_matrix_metrics}, {**dev_stats, **dev_matrix_metrics}


def run_probing_experiment(model, train_loader, dev_loader, n_layers, hook_name="resid_post", pooling="last"):
    """
    Запускает probing по всем слоям за один проход
    Собирает: accuracy, статистику активаций, информацию о классификаторах
    """
    results = {
        "train_accs": [],
        "dev_accs": [],
        "train_stats": [],  # список словарей со статистикой
        "dev_stats": [],
        "layers": []
    }

    for layer in tqdm(range(n_layers), desc="Probing layers"):
        train_acc, dev_acc, train_stats, dev_stats = run_probing_for_layer(
            model, train_loader, dev_loader, layer, hook_name, pooling
        )

        results["train_accs"].append(train_acc)
        results["dev_accs"].append(dev_acc)
        results["train_stats"].append(train_stats)
        results["dev_stats"].append(dev_stats)
        results["layers"].append(layer)

        torch.cuda.empty_cache()

    return results

def load_probing_results(model_size, variant, hook_name="resid_post", pooling="last", results_dir=None):
    """
    Загружает результаты probing из JSON-файла.
    Возвращает словарь results с ключами: metadata, summary, layers, train_accs, dev_accs, train_stats, dev_stats
    """
    if results_dir is None:
        results_dir = PROJECT_ROOT / "results" / "probing"
    fname = f"probe_{model_size}_{variant}_{hook_name}_{pooling}.json"
    fpath = results_dir / fname
    if not fpath.exists():
        raise FileNotFoundError(f"Файл {fpath} не найден. Сначала запустите эксперимент.")
    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data