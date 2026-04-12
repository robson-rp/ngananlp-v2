---
base_model: facebook/nllb-200-distilled-1.3B
library_name: peft
language:
- por
- lin
- umb
- kmb
- cjk
- lue
tags:
- base_model:adapter:facebook/nllb-200-distilled-1.3B
- lora
- translation
- low-resource
- african-languages
- bantu
- transformers
metrics:
- bleu
- chrf
model-index:
- name: ngananlp-v2
  results:
  - task:
      type: translation
    dataset:
      name: NganaNLP test set
      type: custom
    metrics:
    - name: BLEU (overall)
      type: bleu
      value: 24.26
    - name: chrF (overall)
      type: chrf
      value: 48.71
    - name: chrF por_Latn → lin_Latn
      type: chrf
      value: 62.41
    - name: chrF lin_Latn → umb_Latn
      type: chrf
      value: 62.15
    - name: chrF por_Latn → umb_Latn
      type: chrf
      value: 61.45
    - name: chrF kmb_Latn → lin_Latn
      type: chrf
      value: 60.27
    - name: chrF kmb_Latn → umb_Latn
      type: chrf
      value: 59.04
    - name: chrF lue_Latn → lin_Latn
      type: chrf
      value: 58.04
    - name: chrF por_Latn → lue_Latn
      type: chrf
      value: 28.81
---

# NganaNLP v2 — Multilingual Translation for Angolan Languages

LoRA adapter fine-tuned on top of [facebook/nllb-200-distilled-1.3B](https://huggingface.co/facebook/nllb-200-distilled-1.3B) for multilingual translation across 6 languages spoken in Angola and the broader Central African region.

## Model Details

### Model Description

NganaNLP v2 enables translation between Portuguese and four Bantu languages — Lingala, Umbundu, Kimbundu, and Tchokwe — as well as Luvale. The model was fine-tuned using LoRA (Low-Rank Adaptation) to minimise compute cost while adapting NLLB-200 to a specialised low-resource multilingual setting.

Luvale (`lue_Latn`) was added as a new token since it is absent from the original NLLB-200 vocabulary. Its embedding was initialised from the average of three linguistically close proxy languages: Tchokwe (`cjk_Latn`), Lingala (`lin_Latn`), and Umbundu (`umb_Latn`).

- **Developed by:** Robson Paulo
- **Model type:** Seq2Seq (LoRA adapter over NLLB-200-distilled-1.3B)
- **Languages:** Portuguese (`por_Latn`), Lingala (`lin_Latn`), Umbundu (`umb_Latn`), Kimbundu (`kmb_Latn`), Tchokwe (`cjk_Latn`), Luvale (`lue_Latn`)
- **License:** cc-by-nc-4.0
- **Finetuned from:** [facebook/nllb-200-distilled-1.3B](https://huggingface.co/facebook/nllb-200-distilled-1.3B)

### Model Sources

- **Repository:** [robsonrtp/ngananlp-v2](https://huggingface.co/robsonrtp/ngananlp-v2)

## Uses

### Direct Use

This model is intended for translating between any of the 6 supported languages. It is suitable for content localisation, digital inclusion tools, and NLP research on under-resourced Bantu languages.

### Downstream Use

Can be integrated into translation APIs, mobile apps, or content management systems targeting Angolan and Central African language communities.

### Out-of-Scope Use

- **Luvale as a target language** (`X → lue_Latn`) is not recommended — the model was not able to learn reliable generation for this language due to limited training data and the token being absent from the original NLLB vocabulary. Use Luvale as a **source** language only.
- General-purpose translation outside the 6 supported languages.
- High-stakes applications (legal, medical) without human review.

## Bias, Risks, and Limitations

- Training data consists primarily of religious texts (Jehovah's Witnesses publications), which may bias the model towards that domain and vocabulary.
- Tchokwe (`cjk_Latn`) output may occasionally mix in Luvale vocabulary due to noise in the original NLLB-200 training data for that language code.
- Low-resource directions (e.g. `X → kmb_Latn`, `X → cjk_Latn`) have lower scores and should be used with caution in production.

### Recommendations

Test the model on your target domain before deployment. For best results, use the higher-scoring directions listed in the evaluation table below.

## How to Get Started with the Model

```python
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

MODEL_NAME = "facebook/nllb-200-distilled-1.3B"
ADAPTER    = "robsonrtp/ngananlp-v2"

tokenizer  = AutoTokenizer.from_pretrained(ADAPTER)
base_model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME, dtype=torch.bfloat16, low_cpu_mem_usage=True
)
base_model.resize_token_embeddings(len(tokenizer))
model = PeftModel.from_pretrained(base_model, ADAPTER).eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model  = model.to(device)

def translate(text, src_lang, tgt_lang, num_beams=4):
    tokenizer.src_lang = src_lang
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=192).to(device)
    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=256,
            num_beams=num_beams,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
        )
    return tokenizer.batch_decode(output, skip_special_tokens=True)[0]

print(translate("Como pode aumentar o seu amor por Jeová?", "por_Latn", "kmb_Latn"))
# => "kyebhi ki utena kukolesa o henda yé kwa jihova?"
```

## Training Details

### Training Data

- ~321,000 parallel sentence pairs (train), ~40,000 (val), ~40,000 (test)
- 30 bidirectional translation directions across 6 languages
- ~10,000–13,000 examples per direction
- 4 fields: `src_lang`, `tgt_lang`, `src`, `tgt`
- Domain: religious/educational texts

### Training Procedure

#### Training Hyperparameters

- **Base model:** facebook/nllb-200-distilled-1.3B
- **Method:** LoRA (Low-Rank Adaptation)
- **LoRA rank:** 16, alpha: 32, dropout: 0.05
- **Target modules:** `q_proj`, `v_proj`
- **Training regime:** bf16 mixed precision
- **Batch size:** 16 (effective, with gradient accumulation)
- **Learning rate:** 5e-4with linear warmup
- **Max sequence length:** 192 (input), 256 (output)
- **Epochs:** up to 10 with early stopping (patience=3)
- **Evaluation metric:** teacher-forced chrF (predict_with_generate=False)

#### Speeds, Sizes, Times

- **Hardware:** NVIDIA RTX A5000 (24 GB VRAM)
- **Cloud Provider:** RunPod
- **Training time:** ~12 hours
- **Adapter size:** ~1.09 GB (safetensors)

## Evaluation

### Testing Data

1,000 randomly sampled pairs from the held-out test set, covering all 30 translation directions.

### Metrics

- **BLEU** (SacreBLEU) — standard MT metric
- **chrF** — character n-gram F-score, more reliable for morphologically rich languages

### Results

**Overall: BLEU 24.26 / chrF 48.71**

| Direction | BLEU | chrF | N |
|---|---:|---:|---:|
| por_Latn → lin_Latn | 36.37 | 62.41 | 25 |
| umb_Latn → lin_Latn | 36.65 | 62.38 | 26 |
| cjk_Latn → lin_Latn | 37.09 | 62.31 | 31 |
| lin_Latn → umb_Latn | 38.15 | 62.15 | 33 |
| por_Latn → umb_Latn | 31.75 | 61.45 | 28 |
| cjk_Latn → umb_Latn | 32.77 | 61.04 | 30 |
| kmb_Latn → lin_Latn | 31.38 | 60.27 | 47 |
| kmb_Latn → umb_Latn | 34.48 | 59.04 | 33 |
| lue_Latn → lin_Latn | 30.37 | 58.04 | 19 |
| umb_Latn → por_Latn | 31.62 | 54.71 | 29 |
| por_Latn → kmb_Latn | 22.88 | 54.08 | 39 |
| lin_Latn → por_Latn | 32.84 | 54.01 | 24 |
| lue_Latn → umb_Latn | 26.37 | 53.59 | 37 |
| lin_Latn → kmb_Latn | 23.92 | 53.13 | 41 |
| umb_Latn → cjk_Latn | 22.48 | 52.52 | 26 |
| umb_Latn → kmb_Latn | 22.85 | 52.07 | 44 |
| cjk_Latn → por_Latn | 31.20 | 51.53 | 38 |
| kmb_Latn → por_Latn | 28.85 | 50.68 | 26 |
| cjk_Latn → kmb_Latn | 21.49 | 50.51 | 42 |
| lue_Latn → por_Latn | 25.88 | 48.47 | 37 |
| por_Latn → cjk_Latn | 15.19 | 47.47 | 32 |
| lue_Latn → kmb_Latn | 18.40 | 46.97 | 28 |
| lin_Latn → cjk_Latn | 17.43 | 46.92 | 27 |
| lue_Latn → cjk_Latn | 14.44 | 44.04 | 32 |
| kmb_Latn → cjk_Latn | 13.65 | 43.69 | 30 |
| cjk_Latn → lue_Latn | 10.23 | 34.20 | 34 |
| lin_Latn → lue_Latn | 8.23 | 31.07 | 40 |
| por_Latn → lue_Latn | 7.71 | 28.81 | 36 |
| umb_Latn → lue_Latn | 8.33 | 28.74 | 46 |
| kmb_Latn → lue_Latn | 8.72 | 28.59 | 40 |

> ⚠️ All `X → lue_Latn` directions show low scores (chrF < 35) and are **not recommended** for production use. Luvale as a **source** language performs reasonably well (chrF 44–58).

## Environmental Impact

- **Hardware:** NVIDIA RTX A5000 24GB
- **Hours used:** ~12
- **Cloud Provider:** RunPod
- **Compute Region:** US

## Technical Specifications

### Model Architecture and Objective

LoRA adapter over the encoder-decoder architecture of NLLB-200-distilled-1.3B. LoRA layers are applied to the attention query and value projections across all transformer layers.

One new token (`lue_Latn`) was added to the tokenizer and its embedding initialised from the mean of `cjk_Latn`, `lin_Latn`, and `umb_Latn` embeddings.

### Compute Infrastructure

- **Hardware:** NVIDIA RTX A5000 (24 GB VRAM)
- **Software:** PyTorch 2.x, Transformers ≥ 4.40, PEFT 0.18.1, bitsandbytes

### Framework versions

- PEFT 0.18.1