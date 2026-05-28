# Утилиты для сбора активаций и обучения SAE

import torch
import gc
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm


class TopKSAE(torch.nn.Module):
    """Top-K SAE без нормализации"""
    def __init__(self, d_in, d_sae, k):
        super().__init__()
        self.k = k
        # Инициализация весов
        self.W_enc = torch.nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_in, d_sae)))
        self.W_dec = torch.nn.Parameter(torch.nn.init.kaiming_uniform_(torch.empty(d_sae, d_in)))
        self.b_enc = torch.nn.Parameter(torch.zeros(d_sae))
        self.b_dec = torch.nn.Parameter(torch.zeros(d_in))
        # Нормализация столбцов декодера при инициализации
        self.W_dec.data = self.W_dec.data / self.W_dec.data.norm(dim=0, keepdim=True)

    def encode(self, x):
        pre = x @ self.W_enc + self.b_enc
        values, indices = torch.topk(pre, self.k, dim=-1)
        sparse = torch.zeros_like(pre)
        sparse.scatter_(-1, indices, values)
        return sparse

    def decode(self, sparse):
        return sparse @ self.W_dec + self.b_dec

    def forward(self, x):
        sparse = self.encode(x)
        recon = self.decode(sparse)
        return sparse, recon


def collect_all_activations(dataset, model, tokenizer, hook_point, max_length,
                           collect_batch_size=4, prepend_bos=True, device="cuda"):
    """
    Предсобирает все активации датасета в один тензор на CPU
    """
    model.eval()
    all_acts = []

    print(f"Сбор активаций: {len(dataset)} примеров, batch_size={collect_batch_size}")
    for i in tqdm(range(0, len(dataset), collect_batch_size), desc="Collecting"):
        batch_texts = [dataset[j]["text"] for j in range(i, min(i + collect_batch_size, len(dataset)))]

        # Токенизация на CPU для экономии VRAM
        tokens = model.to_tokens(batch_texts, prepend_bos=prepend_bos).cpu()
        if max_length is not None and tokens.shape[1] > max_length:
            tokens = tokens[:, :max_length]
        tokens = tokens.to(device)

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=lambda n: n == hook_point)
            acts = cache[hook_point]  # [batch, seq_len, d_model]
            # Извлекаем активацию последнего токена каждого промпта
            lengths = [len(tokenizer.encode(t, add_special_tokens=False)) for t in batch_texts]
            selected = torch.stack([acts[j, min(lengths[j], acts.shape[1]-1), :] for j in range(len(batch_texts))])
            all_acts.append(selected.cpu())  # Сразу на CPU
            # Очистка кэша
            del cache, acts, tokens, selected

        torch.cuda.empty_cache()
        gc.collect()

    # Объединение в один тензор
    result = torch.cat(all_acts, dim=0).to(torch.float32)  # [N, d_model]
    print(f"Активации собраны: {result.shape} | Память: {result.element_size() * result.nelement() / 1024**2:.1f} MB")
    return result


def create_activation_dataloader(activations_tensor, batch_size=128, shuffle=True):
    """Создаёт DataLoader для обучения SAE из предсобранных активаций."""
    dataset = TensorDataset(activations_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)

def save_activations(activations_tensor, save_path, metadata=None):
    """
    Сохраняет тензор активаций на диск с метаданными.

    Args:
        activations_tensor: torch.Tensor [N, d_model]
        save_path: str или Path, путь к файлу .pt
        metadata: dict, дополнительная информация (слой, модель, дата)
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    save_data = {
        "activations": activations_tensor,
        "metadata": metadata or {},
        "shape": activations_tensor.shape,
        "dtype": str(activations_tensor.dtype)
    }
    torch.save(save_data, save_path)
    print(f"Активации сохранены: {save_path} | Размер: {save_path.stat().st_size / 1024**2:.1f} MB")
    return save_path


def load_activations(load_path, device="cuda"):
    """
    Загружает предсобранные активации с диска.

    Returns:
        activations_tensor: torch.Tensor [N, d_model] на указанном устройстве
        metadata: dict с информацией о сборе
    """
    load_path = Path(load_path)
    if not load_path.exists():
        raise FileNotFoundError(f"Файл активаций не найден: {load_path}")

    data = torch.load(load_path, map_location="cpu", weights_only=False)
    activations = data["activations"].to(device)
    metadata = data.get("metadata", {})

    print(f"Активации загружены: {data['shape']} | Устройство: {device}")
    return activations, metadata