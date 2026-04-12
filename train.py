# train.py - ngananlp-v2
# Correções: tokenização decoder, beams=4, AdamW, lue_Latn init,
#            normalização de texto, rebalanceamento de publicações

# ─── Instalação ────────────────────────────────────────────────────
import importlib, subprocess, os

def _need_install(import_name: str) -> bool:
    try:
        importlib.import_module(import_name)
        return False
    except ImportError:
        return True

_missing = [
    pkg for pkg, mod in {
        "transformers": "transformers",
        "peft":         "peft",
        "evaluate":     "evaluate",
        "sacrebleu":    "sacrebleu",
        "sentencepiece":"sentencepiece",
        "protobuf":     "google.protobuf",
    }.items() if _need_install(mod)
]

if _missing:
    print(f"A instalar pacotes em falta: {_missing}")
    subprocess.run(["pip", "install", "-q"] + _missing, check=False)
else:
    print("Todos os pacotes já instalados — a saltar pip install.")

# ─── Imports ───────────────────────────────────────────────────────
import json
import random
import unicodedata
import torch
import evaluate
import numpy as np
from collections import defaultdict
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from peft import get_peft_model, LoraConfig, TaskType

# ─── Config ────────────────────────────────────────────────────────
MODEL_NAME   = "facebook/nllb-200-distilled-1.3B"
OUTPUT_DIR   = os.getenv("OUTPUT_DIR", "checkpoints")
LORA_DIR     = os.getenv("LORA_DIR",   "lora_adapter")
DATA_DIR     = os.getenv("DATA_DIR",   "data")

MAX_LEN      = 192
GEN_MAX_LEN  = 256
BATCH_SIZE   = 8
EVAL_BATCH   = 8
GRAD_ACCUM   = 4        # effective batch = 32
EPOCHS       = 6
LR           = 3e-5
WARMUP_RATIO = 0.10
LORA_R       = 16
LORA_ALPHA   = 32
EVAL_SAMPLES = 3000
SEED         = 42

# Pesos de rebalanceamento por publicação
# Reduz dominância de 'w' (94%) e amplifica registos minoritários
PUBLICATION_WEIGHTS = {
    "w":  0.30,   # Sentinela        — reduzir de 94% para 30%
    "bh": 0.20,   # Livro didático   — amplificar
    "fg": 0.20,   # Evangelístico    — amplificar
    "jl": 0.15,   # Informativo      — amplificar
    "th": 0.10,   # Instrucional     — amplificar
    "yc": 0.05,   # Coloquial/jovem  — amplificar de 0.4% para 5%
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LORA_DIR,   exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 60)
print(f"Model:           {MODEL_NAME}")
print(f"Output:          {OUTPUT_DIR}")
print(f"Data:            {DATA_DIR}")
print(f"LR:              {LR}")
print(f"Epochs:          {EPOCHS}")
print(f"Effective batch: {BATCH_SIZE * GRAD_ACCUM}")
if torch.cuda.is_available():
    print(f"GPU:             {torch.cuda.get_device_name(0)}")
    print(f"VRAM:            {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("GPU:             not available — a correr em CPU")
print("=" * 60)

# ─── Normalização de texto ─────────────────────────────────────────
def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)  # diacríticos bantu (ã, õ, etc.)
    text = text.lower()                         # uppercase → lowercase consistente
    text = " ".join(text.split())              # espaços múltiplos
    return text

# ─── Rebalanceamento de publicações ────────────────────────────────
def rebalance_dataset(data: list, weights: dict, seed: int = SEED) -> list:
    """
    Reamostrar o dataset para respeitar os pesos por publicação.
    Publicações sub-representadas são oversampled (com repetição).
    Publicações sobre-representadas são undersampled.
    Exemplos sem campo 'publication' são mantidos tal como estão.
    """
    rng = random.Random(seed)

    by_pub = defaultdict(list)
    no_pub = []
    for example in data:
        pub = example.get("publication", None)
        if pub:
            by_pub[pub].append(example)
        else:
            no_pub.append(example)

    if not by_pub:
        print("Campo 'publication' não encontrado — a usar dataset original sem rebalanceamento")
        return data

    total  = len(data)
    result = []
    for pub, w in weights.items():
        if pub not in by_pub:
            print(f"  Publicação '{pub}' não encontrada no dataset — a ignorar")
            continue
        examples = by_pub[pub]
        target_n = int(total * w)
        if target_n <= len(examples):
            sampled = rng.sample(examples, target_n)
        else:
            sampled  = examples * (target_n // len(examples))
            sampled += rng.sample(examples, target_n % len(examples))
        result.extend(sampled)
        print(f"  {pub}: {len(examples):>6} → {len(sampled):>6} ({w*100:.0f}%)")

    for pub in set(by_pub.keys()) - set(weights.keys()):
        print(f"  {pub}: sem peso definido — a manter {len(by_pub[pub])} exemplos")
        result.extend(by_pub[pub])

    result.extend(no_pub)
    rng.shuffle(result)
    print(f"  Total: {len(data)} → {len(result)} exemplos após rebalanceamento\n")
    return result

# ─── Carregamento do dataset ───────────────────────────────────────
def load_jsonl(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

print("\nLoading dataset...")
train_raw = load_jsonl(f"{DATA_DIR}/train.json")
val_raw   = load_jsonl(f"{DATA_DIR}/val.json")
test_raw  = load_jsonl(f"{DATA_DIR}/test.json")

print(f"  Treino original:    {len(train_raw)}")
print(f"  Validação original: {len(val_raw)}")
print(f"  Teste original:     {len(test_raw)}")

# Rebalancear apenas o treino — val e test mantêm distribuição real
print("\nRebalanceando dataset de treino:")
train_raw = rebalance_dataset(train_raw, PUBLICATION_WEIGHTS)

dataset = DatasetDict({
    "train":      Dataset.from_list(train_raw),
    "validation": Dataset.from_list(val_raw),
    "test":       Dataset.from_list(test_raw),
})
print(dataset)

# ─── Tokenizer ─────────────────────────────────────────────────────
print("\nLoading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

new_tokens = ["lue_Latn"]  # Luvale ausente no NLLB base
added = tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
print(f"Added {added} new token(s): {new_tokens}")

# ─── Model ─────────────────────────────────────────────────────────
print("Loading model...")

# T4 não suporta bfloat16 — usar float16
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
print(f"dtype: {dtype}")

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True,
)
model.resize_token_embeddings(len(tokenizer))
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

# ─── Inicializar embedding lue_Latn com média de línguas próximas ──
# cjk_Latn = Chokwe (ISO 639-3: cjk) — língua mais próxima do Luvale no NLLB
# lin_Latn = Lingala, umb_Latn = Umbundu — outras línguas bantu da região
with torch.no_grad():
    candidates = ["cjk_Latn", "lin_Latn", "umb_Latn"]
    unk_id = tokenizer.unk_token_id

    valid = []
    for t in candidates:
        tid = tokenizer.convert_tokens_to_ids(t)
        if tid != unk_id:
            valid.append((t, tid))
            print(f"  ✓ {t:12s} → id {tid}")
        else:
            print(f"  ✗ {t:12s} → não encontrado no vocabulário NLLB")

    if valid:
        valid_ids = [tid for _, tid in valid]
        mean_emb  = model.model.shared.weight[valid_ids].mean(dim=0)
        new_id    = tokenizer.convert_tokens_to_ids("lue_Latn")
        model.model.shared.weight[new_id] = mean_emb
        if not model.config.tie_word_embeddings:
            model.lm_head.weight[new_id] = mean_emb
        used = [t for t, _ in valid]
        print(f"lue_Latn embedding inicializado a partir de: {used}")
    else:
        print("Warning: nenhum token proxy encontrado — lue_Latn com embedding aleatório")

print("Model ready.")

# ─── LoRA ──────────────────────────────────────────────────────────
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "v_proj", "k_proj", "out_proj",
        "encoder_attn.q_proj", "encoder_attn.v_proj",
        "encoder_attn.k_proj", "encoder_attn.out_proj",
    ],
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ─── Tokenização ───────────────────────────────────────────────────
# as_target_tokenizer() foi removido no transformers>=4.40.
# Equivalente moderno: tokenizer.src_lang = tgt_lang antes de tokenizar o target.
def preprocess(examples):
    all_input_ids, all_attention_mask, all_labels = [], [], []

    for src_lang, tgt_lang, src, tgt in zip(
        examples["src_lang"],
        examples["tgt_lang"],
        examples["src"],
        examples["tgt"],
    ):
        src = normalize_text(src)
        tgt = normalize_text(tgt)

        # --- encoder ---
        tokenizer.src_lang = src_lang
        enc = tokenizer(
            src,
            max_length=MAX_LEN,
            truncation=True,
            padding="max_length",
        )

        # --- decoder / labels ---
        # src_lang = tgt_lang faz o NLLB antepor o token de língua correcto nos labels
        tokenizer.src_lang = tgt_lang
        dec = tokenizer(
            tgt,
            max_length=GEN_MAX_LEN,
            truncation=True,
            padding="max_length",
        )

        labels = [
            tok if tok != tokenizer.pad_token_id else -100
            for tok in dec["input_ids"]
        ]

        all_input_ids.append(enc["input_ids"])
        all_attention_mask.append(enc["attention_mask"])
        all_labels.append(labels)

    return {
        "input_ids":      all_input_ids,
        "attention_mask": all_attention_mask,
        "labels":         all_labels,
    }

print("\nTokenizing...")
remove_cols = ["src_lang", "tgt_lang", "src", "tgt"]
if "publication" in dataset["train"].column_names:
    remove_cols.append("publication")

tokenized = dataset.map(
    preprocess,
    batched=True,
    batch_size=64,
    remove_columns=remove_cols,
    num_proc=1,
)
tokenized["validation"] = tokenized["validation"].select(
    range(min(EVAL_SAMPLES, len(tokenized["validation"])))
)
print(tokenized)

# ─── Métricas ──────────────────────────────────────────────────────
bleu_metric = evaluate.load("sacrebleu")
chrf_metric = evaluate.load("chrf")

def preprocess_logits_for_metrics(logits, labels):
    # Seq2SeqTrainer pode devolver tuplo (logits, past_key_values, ...)
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # predictions já são token IDs (argmax feito em preprocess_logits_for_metrics)
    decoded_preds  = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels         = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds  = [p.strip().lower() for p in decoded_preds]
    decoded_labels = [l.strip().lower() for l in decoded_labels]

    bleu = bleu_metric.compute(
        predictions=decoded_preds,
        references=[[l] for l in decoded_labels],
    )
    chrf = chrf_metric.compute(
        predictions=decoded_preds,
        references=decoded_labels,
    )
    return {
        "bleu": round(bleu["score"], 2),
        "chrf": round(chrf["score"], 2),
    }

# ─── Training Args ─────────────────────────────────────────────────
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,

    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,

    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="chrf",
    greater_is_better=True,
    save_total_limit=2,

    learning_rate=LR,
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type="cosine",
    weight_decay=0.01,

    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    optim="adamw_torch",
    predict_with_generate=False,  # multilingual: sem forced_bos_token_id único possível
                                  # chrF calculado por argmax nos logits (teacher-forced)

    logging_steps=50,
    report_to="none",
    dataloader_num_workers=4,
)

# ─── Trainer ───────────────────────────────────────────────────────
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
    pad_to_multiple_of=8,
    label_pad_token_id=-100,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

# ─── Treino ────────────────────────────────────────────────────────
print(f"\nTrain pairs:      {len(tokenized['train'])}")
print(f"Validation pairs: {len(tokenized['validation'])}")
steps_per_epoch = len(tokenized["train"]) // (BATCH_SIZE * GRAD_ACCUM)
print(f"Steps per epoch:  {steps_per_epoch}")
print(f"Total steps:      {steps_per_epoch * EPOCHS}\n")

trainer.train()

# ─── Guardar ───────────────────────────────────────────────────────
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# Adaptador LoRA separado — muito mais leve para descarregar (~50MB)
model.save_pretrained(LORA_DIR)
tokenizer.save_pretrained(LORA_DIR)

print(f"\nDone.")
print(f"Modelo completo:  {OUTPUT_DIR}")
print(f"Adaptador LoRA:   {LORA_DIR}")