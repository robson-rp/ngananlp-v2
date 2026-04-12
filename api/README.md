---
title: NganaNLP v2
emoji: 🌍
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# NganaNLP v2 — Translation API

REST API for multilingual translation between Portuguese and Angolan Bantu languages (Lingala, Umbundu, Kimbundu, Tchokwe, Luvale).

**Docs:** `GET /docs`

**Translate:** `POST /translate`
```json
{
  "text": "Como tomar boas decisões sobre educação?",
  "src_lang": "por_Latn",
  "tgt_lang": "kmb_Latn"
}
```
