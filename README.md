# Symbolic Rule Extraction from LLM Hidden Representations

Код и экспериментальные ноутбуки магистерской диссертации «Метод извлечения символьных правил и причинно-следственных связей из скрытых представлений языковой модели»

## Краткое описание

Работа посвящена механистической интерпретируемости больших языковых моделей (LLM). Основная идея - декомпозиция скрытых представлений трансформера с помощью разреженного автокодировщика (SAE) и последующее извлечение символьных правил (пороговых условий, деревьев решений, ансамблей RuleFit), аппроксимирующих поведение модели. Полученные правила проходят каузальную верификацию через интервенционный анализ (абляция и патчинг).


## Структура репозитория
```text
├── src/ # Основные модули
│ ├── data.py # Загрузка данных и моделей
│ ├── probing.py # Probing classifiers
│ ├── logit_lens.py # Logit Lens
│ ├── baseline_rules.py # Baseline на сырых нейронах
│ ├── sae.py # Обучение Top‑K SAE
│ ├── rule_extraction.py# Извлечение правил из латентов
│ ├── interventions.py # Интервенционный анализ
│ ├── llm_upgrade.py # Дообучение моделей (QLoRA)
│ └── dataset_cot.py # Подготовка датасетов в CoT/прямом формате
├── data/ # RuleTaker (не включён в репозиторий)
├── results/ # Веса моделей, метрики (не включены в репозиторий)
├── config/ # Конфигурационные YAML‑файлы (опционально)
├── notebooks/ # Jupyter‑ноутбуки с экспериментами
│ ├── 00_check_setup.ipynb
│ ├── 01_data_setup.ipynb
│ ├── 02_layer_probing_*.ipynb
│ ├── 03_logit_lens_*.ipynb
│ ├── 04_baseline_*.ipynb
│ ├── 05_sae_training_*.ipynb
│ ├── 06_rule_extraction_*.ipynb
│ └── 07_interventions_*.ipynb
├── requirements.txt # Список зависимостей
├── pyproject.toml # Метаданные проекта
└── README.md
```

## Установка и зависимости

### С использованием pip и requirements.txt

```bash
python -m venv .venv
source .venv/bin/activate  # или .venv\Scripts\activate на Windows
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
```

### С использованием `uv`

```bash
# Клонирование репозитория
git clone https://github.com/martetten/Symbolic-rule-extraction.git
cd Symbolic-rule-extraction

# Создание виртуального окружения и установка
uv venv
uv sync
```

## Данные

Эксперименты используют синтетический датасет RuleTaker (Allen Institute for AI). Скачайте его и поместите в папку data/raw/rultaker/rule-reasoning-dataset-V2020.2.5.0/problog/

Ссылка на оригинальный датасет: https://github.com/allenai/ruletaker

После загрузки структура должна быть такой:
```text
data/raw/rultaker/rule-reasoning-dataset-V2020.2.5.0/problog/
├── depth-0/
│   ├── train.jsonl
│   ├── dev.jsonl
│   └── test.jsonl
├── depth-1/
│   └── ...
├── depth-2/
└── ...
```

## Воспроизведение результатов

Основные этапы пайплайна выполняются в следующем порядке:

- 00_check_setup.ipynb – проверка оборудования и моделей.
- 01_data_setup.ipynb – загрузка и предобработка RuleTaker.
- 01_llm_finetune_*.ipynb – дообучение моделей через QLoRA.
- 02_layer_probing_*.ipynb – зондирование слоёв и выбор оптимального слоя.
    - exp1 - Qwen2.5-1.5b (тупиковая, модель не участвует в дальнейших экспериментах)
    - exp2 - Pythia410m
    - exp3 - Pythia-1b (exp3-2 QLoRA (full depth-0), exp8-2 seq QLoRA (full depth-1))
    - exp4, 5 - Pythia-1b (различные неудачные попытки добиться достаточной точности на depth-1)
    - exp6 - GPT-2 Large (exp6-1 QLoRA (full depth-0), exp7-2 seq QLoRA (full depth-1))
- 03_logit_lens_*.ipynb – анализ через Logit Lens (опционально).
- 04_baseline_*.ipynb – построение базовых линий на нейронах.
- 05_sae_training_*.ipynb – обучение SAE на выбранном слое.
- 06_rule_extraction_*.ipynb – извлечение правил из латентов и сравнение с baseline.
- 06_interventions_*.ipynb – каузальная верификация через абляцию и патчинг.

Рекомендуется запускать их последовательно для каждой конфигурации expN