import json
import os
import torch
import evaluate as hf_evaluate
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

MODEL_NAME = "facebook/nllb-200-distilled-1.3B"
HUB_REPO   = "robsonrtp/ngananlp-v2"           # fallback se LORA_DIR não existir
LORA_DIR   = os.getenv("LORA_DIR",  "lora_adapter")
DATA_DIR   = os.getenv("DATA_DIR",  "data")

# Usa adaptador local se existir, caso contrário carrega do Hub
adapter_source = LORA_DIR if os.path.isdir(LORA_DIR) else HUB_REPO
print(f"Adapter source: {adapter_source}", flush=True)

# ─── 1. Tokenizador ───────────────────────────────────────────────
print("Loading tokenizer...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(adapter_source)
print(f"Vocab size: {len(tokenizer)}", flush=True)

# ─── 2. Carrega modelo ────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    free, total = torch.cuda.mem_get_info()
    print(f"GPU VRAM: {free/1e9:.1f} GB free / {total/1e9:.1f} GB total", flush=True)

print(f"Loading model... (device={device})", flush=True)
try:
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    print("Base model loaded. Resizing token embeddings...", flush=True)
    base_model.resize_token_embeddings(len(tokenizer))
    print(f"Embeddings resized. Moving model to {device}...", flush=True)
    base_model = base_model.to(device)
    if device == "cuda":
        torch.cuda.synchronize()
        free_after, _ = torch.cuda.mem_get_info()
        print(f"Model on device. VRAM used: {(total - free_after)/1e9:.1f} GB", flush=True)
    print(f"Loading PEFT adapter from {adapter_source}...", flush=True)
    model = PeftModel.from_pretrained(base_model, adapter_source)
    print("PEFT adapter loaded. Setting eval mode...", flush=True)
    model.eval()
    print("Model ready.", flush=True)
except RuntimeError as e:
    print(f"\n[ERROR] {e}", flush=True)
    raise




# ─── 3. Função de tradução ─────────────────────────────────────────
def traduzir(texto, src_lang, tgt_lang, num_beams=4):
    tokenizer.src_lang = src_lang
    inputs = tokenizer(
        texto, return_tensors="pt", padding=True, truncation=True, max_length=192
    ).to(device)
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

def traduzir_batch(textos, src_lang, tgt_lang, num_beams=4):
    tokenizer.src_lang = src_lang
    inputs = tokenizer(
        textos, return_tensors="pt", padding=True, truncation=True, max_length=192
    ).to(device)
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
    return tokenizer.batch_decode(output, skip_special_tokens=True)

# ─── 4. Avaliação qualitativa ──────────────────────────────────────
print("\n" + "="*60, flush=True)
print("QUALITATIVE EVALUATION — num_beams=4", flush=True)
print("="*60, flush=True)

pares = [
    # por_Latn -> umb_Latn
    ("por_Latn", "umb_Latn", "'SERÁ QUE EU AINDA SOU ÚTIL PARA JEOVÁ?'"),
    # => "ANGA HẼ CILO HANDI NDI KUETE ESILIVILO KU YEHOVA?"

    ("por_Latn", "umb_Latn", "Mostre perspicácia e 'seja bem-sucedido'"),
    # => "Lekisa Olondunge Kuenje 'o ka Kuata Onima Yiwa'"

    # por_Latn -> kmb_Latn
    ("por_Latn", "kmb_Latn", "Mostre perspicácia e 'seja bem-sucedido'"),
    # => "Kala u Muthu wa Tetuluka Anga 'u Zediwa ku Mwenyu'"

    ("por_Latn", "kmb_Latn", "Como pode aumentar o seu amor por Jeová?"),
    # => "Kyebhi ki utena kubandekesa o henda ye kwa Jihova?"

    # por_Latn -> cjk_Latn (Tchokwe)
    ("por_Latn", "cjk_Latn", "Como tomar boas decisões sobre educação?"),
    # => "Nyonga Kanawa Shimbu Kanda Uchiya ku Shikola Yinene"

    ("por_Latn", "cjk_Latn", "COMO PODEMOS MOSTRAR A NOSSA GRATIDÃO..."),
    # => "KUCHI MUTUHASA KUSOLOLA NGWETU TWAKUSAKWILILA..."

    # por_Latn -> lue_Latn
    ("por_Latn", "lue_Latn", "'SERÁ QUE EU AINDA SOU ÚTIL PARA JEOVÁ?'"),
    # => "KACHI NGE MULI NAKUHUHWASANA NUMBA NGE YEHOVA NAHASE KUMIZACHISA"

    ("por_Latn", "lue_Latn", "IMITE PAULO E VISTA A NOVA PERSONALIDADE"),
    # => "TWALENUHO LIKA KUVWALA MUTU WAMUHYA NGANA MWAPAULU"

    # por_Latn -> lin_Latn
    ("por_Latn", "lin_Latn", "Como a Bíblia pode ajudá-lo a perseverar"),
    # => "Ndenge oyo koyekola Biblia ekoki kosalisa yo oyika mpiko na mikakatano"

    ("por_Latn", "lin_Latn", "PORQUE FAZER DISCÍPULOS EXIGE PACIÊNCIA?"),
    # => "MPO NA NINI TOSENGELI KOZALA NA MOTEMA MOLAI?"

    # umb_Latn -> por_Latn
    ("umb_Latn", "por_Latn", "LEKISA ESUNGULUKO ECI EKALO LI PONGOLOKA"),
    # => "SEJA RAZOÁVEL QUANDO AS CIRCUNSTÂNCIAS MUDAM"

    ("umb_Latn", "por_Latn", "Elavoko ka li tu sumuisa.—Va Rom. 5:5."),
    # => ""A esperança não leva à deceção." — ROM. 5:5."

    # kmb_Latn -> por_Latn
    ("kmb_Latn", "por_Latn", "O HENDA YENE I TU SWÍNISA KUYA MU KYÔNGE"),
    # => "ASSISTIR ÀS REUNIÕES MOSTRA O NOSSO AMOR"

    ("kmb_Latn", "por_Latn", "O NZAMBI YA KIDI UKUMBIDILA O IKANENU YÊ"),
    # => "O QUE JEOVÁ TEM FEITO PARA CUMPRIR O SEU PROPÓSITO"

    # lue_Latn -> por_Latn
    ("lue_Latn", "por_Latn", "UNO VAWAVISA VATELA KULIMONA NGACHILIHI?"),
    # => "COMO É QUE OS UNGIDOS DEVEM ENCARAR-SE A SI PRÓPRIOS?"

    ("lue_Latn", "por_Latn", "'Kuli Lwola lwaKuzata naLwola lwaKunoka'"),
    # => ""Há um tempo determinado" para trabalhar e para descansar"

    # umb_Latn -> kmb_Latn
    ("umb_Latn", "kmb_Latn", "Kolela Kohenda 'Yonganji Yongongo Yosi!'"),
    # => "Dyelela ku Mufundixi wa Henda wa Ngongo Yoso!"

    # kmb_Latn -> umb_Latn
    ("kmb_Latn", "umb_Latn", "Jihova o henda yê yadikota.—TIY. 5:11."),
    # => ""Yehova ukuacisola calua haeye ukuahenda."—TIA. 5:11."

    # umb_Latn -> lue_Latn
    ("umb_Latn", "lue_Latn", "Nye ci tu kuatisa oku liwekapo oku pisa?"),
    # => "Vyuma muka navitukafwa tulitwamine kusopesanga vakwetu?"

    # lue_Latn -> umb_Latn
    ("lue_Latn", "umb_Latn", "VYUMA TWATELA KULINGA NGE TULI MULUYANDO"),
    # => "OVINA TU SUKILA OKU LINGA ECI TU LIYAKA LOVITANGI"

    ("lue_Latn", "umb_Latn", "Mwomwo ika twatela kutunga usepa naYesu?"),
    # => "Momo lie tu sukilila oku kala akamba va Yesu?"
]



for src_lang, tgt_lang, texto in pares:
    traducao = traduzir(texto, src_lang, tgt_lang)
    print(f"\n[{src_lang} → {tgt_lang}]", flush=True)
    print(f"  Original: {texto}", flush=True)
    print(f"  Tradução: {traducao}", flush=True)

# ─── 5. Avaliação BLEU/chrF no conjunto de teste ──────────────────
TEST_FILE = os.path.join(DATA_DIR, "test.json")
if os.path.exists(TEST_FILE):
    print("\n" + "="*60, flush=True)
    print("BLEU + chrF — test set (num_beams=4)", flush=True)
    print("="*60, flush=True)

    bleu_metric = hf_evaluate.load("sacrebleu")
    chrf_metric = hf_evaluate.load("chrf")

    def load_jsonl(path):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(l) for l in f]

    test_data = load_jsonl(TEST_FILE)[:1000]  # 1000 pares do teste

    BATCH_SIZE = 16
    from collections import defaultdict

    # Agrupa por (src_lang, tgt_lang) para batch homogéneo
    groups = defaultdict(list)
    for i, ex in enumerate(test_data):
        groups[(ex["src_lang"], ex["tgt_lang"])].append((i, ex["src"], ex["tgt"]))

    preds = [None] * len(test_data)
    refs  = [None] * len(test_data)

    total_groups = len(groups)
    dir_preds = {}  # (src_lang, tgt_lang) -> (preds, refs)
    for g_idx, ((src_lang, tgt_lang), items) in enumerate(groups.items(), 1):
        print(f"  [{g_idx}/{total_groups}] {src_lang} → {tgt_lang} ({len(items)} exemplos)", flush=True)
        dir_p, dir_r = [], []
        for batch_start in range(0, len(items), BATCH_SIZE):
            batch = items[batch_start:batch_start + BATCH_SIZE]
            idxs, srcs, tgts = zip(*batch)
            translations = traduzir_batch(list(srcs), src_lang, tgt_lang)
            for idx, tgt, trans in zip(idxs, tgts, translations):
                preds[idx] = trans
                refs[idx]  = tgt
                dir_p.append(trans)
                dir_r.append(tgt)
        dir_preds[(src_lang, tgt_lang)] = (dir_p, dir_r)

    # ── Scores globais ──
    bleu = bleu_metric.compute(predictions=preds, references=[[r] for r in refs])
    chrf = chrf_metric.compute(predictions=preds, references=refs)

    print(f"\nBLEU:  {round(bleu['score'], 2)}", flush=True)
    print(f"chrF:  {round(chrf['score'], 2)}", flush=True)

    # ── Breakdown por direcção ──
    print("\n" + "="*60, flush=True)
    print("BREAKDOWN POR DIRECÇÃO", flush=True)
    print("="*60, flush=True)
    print(f"{'Direcção':<30} {'BLEU':>6}  {'chrF':>6}  {'N':>4}", flush=True)
    print("-"*50, flush=True)

    dir_results = []
    for (src_lang, tgt_lang), (dp, dr) in dir_preds.items():
        b = bleu_metric.compute(predictions=dp, references=[[r] for r in dr])
        c = chrf_metric.compute(predictions=dp, references=dr)
        dir_results.append((src_lang, tgt_lang, b["score"], c["score"], len(dp)))

    # Ordenar por chrF descendente
    dir_results.sort(key=lambda x: x[3], reverse=True)
    for src_lang, tgt_lang, b, c, n in dir_results:
        label = f"{src_lang} → {tgt_lang}"
        print(f"{label:<30} {b:>6.2f}  {c:>6.2f}  {n:>4}", flush=True)

    print("\nEvaluation complete.", flush=True)
else:
    print(f"\n[INFO] {TEST_FILE} not found — skipping BLEU/chrF evaluation.", flush=True)

