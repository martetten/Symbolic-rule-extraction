# src/llm_upgrade.py
# Функции для дообучения (LoRA) моделей LLM

import torch
import os
import platform
from pathlib import Path
from tqdm import tqdm

from transformer_lens import HookedTransformer
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training, PeftModel

from src.data import PROJECT_ROOT


def create_quantization_config(use_4bit=True):
    """
    Создаёт конфигурацию квантования для экономии vram (QLoRA)
    """
    if not use_4bit:
        return None

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )


def create_lora_config(
    r=8,
    alpha=16,
    dropout=0.05,
    target_modules=None
):
    """
    Создаёт конфигурацию LoRA-адаптера
    """
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False
    )


def prepare_model_for_finetune(
    model_name,
    use_qlora=True,
    device="cuda"
):
    """
    Загружает модель и токенизатор, применяет квантование и LoRA
    Возвращает обработанные модель и токенизатор
    """
    quantization_config = create_quantization_config(use_4bit=use_qlora)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Проверка поддержки flash_attention: только Linux + CUDA
    use_flash_attn = False
    if torch.cuda.is_available() and platform.system() == "Linux":
        try:
            from flash_attn import flash_attn_func
            use_flash_attn = True
        except ImportError:
            pass

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config if use_qlora else None,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="flash_attention_2" if use_flash_attn else "eager"
    )

    if use_qlora:
        model = prepare_model_for_kbit_training(model)

    lora_config = create_lora_config()
    model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    return model, tokenizer


def tokenize_for_finetune(ex, tokenizer, max_length=512):
    """
    Токенизирует один пример для обучения языковому моделированию
    Паддинг добавляется динамически через DataCollator
    """
    return tokenizer(
        ex["text"],
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors="pt"
    )


def create_training_args(
    output_dir,
    epochs=3,
    batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    logging_steps=10,
    save_steps=100,
    use_wandb=False,
    eval_dataset_provided=False
):
    """
    Создаёт аргументы обучения Transformers Trainer
    Оптимизировано под 6gb vram
    """

    learning_rate = float(learning_rate)
    batch_size = int(batch_size)
    gradient_accumulation_steps = int(gradient_accumulation_steps)
    epochs = int(epochs)

    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=logging_steps,
        save_steps=save_steps,
        # save_total_limit=10,
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        optim="paged_adamw_8bit",
        report_to="wandb" if use_wandb else "none",
        remove_unused_columns=False,
        push_to_hub=False,
        eval_strategy="steps" if eval_dataset_provided else "no",
        eval_steps=50 if eval_dataset_provided else None,
        # load_best_model_at_end=True if eval_dataset_provided else False
    )


def train_lora_model(
    model,
    tokenizer,
    train_dataset,
    config,
    eval_dataset=None,
    max_length=512
):
    """
    Запускает процесс дообучения с использованием Transformers Trainer
    Возвращает: trained_model, training_metrics
    """
    # Токенизация с динамическим паддингом
    def tokenize_fn(ex):
        return tokenizer(ex["text"], truncation=True, max_length=max_length, padding=False)

    tokenized_train = train_dataset.map(tokenize_fn, batched=False, remove_columns=train_dataset.column_names)

    if eval_dataset is not None:
        tokenized_eval = eval_dataset.map(tokenize_fn, batched=False, remove_columns=eval_dataset.column_names)
    else:
        tokenized_eval = None

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = create_training_args(
        output_dir=config.get("output_dir", str(PROJECT_ROOT / "results/checkpoints/finetune")),
        epochs=config.get("epochs", 3),
        batch_size=config.get("batch_size", 2),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 8),
        learning_rate=config.get("learning_rate", 2e-4),
        logging_steps=config.get("logging_steps", 10),
        save_steps=config.get("save_steps", 50),
        use_wandb=config.get("use_wandb", False),
        eval_dataset_provided=(eval_dataset is not None)
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator
    )

    train_result = trainer.train()

    return trainer.model, train_result.metrics


def save_finetuned_model(model, tokenizer, save_path):
    """
    Сохраняет только веса адаптера LoRA
    """
    save_path = Path(save_path).resolve()
    save_path.mkdir(parents=True, exist_ok=True)

    # model.save_pretrained создаёт adapter_config.json и weights
    model.save_pretrained(str(save_path))
    tokenizer.save_pretrained(str(save_path))
    print(f"Адаптер сохранён в {save_path}")


def load_finetuned_model(
    base_model_name,
    adapter_path,
    device="cuda",
    use_qlora=True
):
    """
    Загружает базовую модель и применяет сохранённый LoRA-адаптер
    """
    model, tokenizer = prepare_model_for_finetune(
        base_model_name,
        use_qlora=use_qlora,
        device=device
    )

    adapter_path = str(Path(adapter_path).resolve())
    config_path = Path(adapter_path) / "adapter_config.json"

    if not config_path.exists():
        # Автоматический откат на последний Trainer-чекпоинт
        parent = Path(adapter_path).parent
        checkpoints = sorted(parent.glob("checkpoint-*"))
        if checkpoints:
            adapter_path = str(checkpoints[-1].resolve())
            config_path = Path(adapter_path) / "adapter_config.json"
            if not config_path.exists():
                raise FileNotFoundError("adapter_config.json отсутствует во всех чекпоинтах. Запустите save_finetuned_model().")
            print(f"Warning: lora_final пуст. Используем последний чекпоинт: {adapter_path}")
        else:
            raise FileNotFoundError(f"adapter_config.json не найден в {adapter_path}")

    # local_files_only=True запрещает попытку парсинга пути как Hub-репозитория
    model = PeftModel.from_pretrained(model, adapter_path, local_files_only=True)
    return model, tokenizer


def wrap_for_transformer_lens(base_model_name, adapter_path, device="cuda"):
    """
    Загружает файнтюненную модель в HookedTransformer для дальнейшего анализа
    Возвращает: hooked_model, tokenizer
    """

    # Загрузка токенизатора
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch.cuda.empty_cache()

    # Загрузка базовой модели без квантования
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map=None,
        low_cpu_mem_usage=False,
        attn_implementation="eager"  # Избегаем конфликтов с flash_attn на Windows
    )

    base_model = base_model.to(device)

    # Применение LoRA адаптера и сливание весов
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False
    )
    model = model.to(device)
    model = model.merge_and_unload()
    model.eval()

    # Оборачивание в HookedTransformer
    hooked_model = HookedTransformer.from_pretrained(
        base_model_name,
        hf_model=model,
        tokenizer=tokenizer,
        device=device,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        fold_value_biases=False,
        dtype=torch.bfloat16
    )

    return hooked_model, tokenizer