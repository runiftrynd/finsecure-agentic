from __future__ import annotations

from typing import Any

import torch

from model_loader import (
    load_intent_resources,
)


MAX_TOKEN_LENGTH = 128
LOW_CONFIDENCE_THRESHOLD = 0.50


def _get_label(
    id2label: dict,
    label_id: int,
) -> str:
    """
    Mengambil label dari konfigurasi model.
    """

    if label_id in id2label:
        return str(
            id2label[label_id]
        )

    string_id = str(label_id)

    if string_id in id2label:
        return str(
            id2label[string_id]
        )

    return f"LABEL_{label_id}"


def predict_intent(
    text: str,
    top_k: int = 3,
) -> dict[str, Any]:
    """
    Memprediksi intent pesan pengguna.
    """

    clean_text = str(text).strip()

    if not clean_text:
        raise ValueError(
            "Teks pertanyaan tidak boleh kosong."
        )

    tokenizer, model, device = (
        load_intent_resources()
    )

    encoded_input = tokenizer(
        clean_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_TOKEN_LENGTH,
    )

    encoded_input = {
        key: value.to(device)
        for key, value
        in encoded_input.items()
    }

    with torch.inference_mode():
        output = model(
            **encoded_input
        )

        probabilities = torch.softmax(
            output.logits[0],
            dim=-1,
        )

    available_labels = (
        model.config.num_labels
    )

    top_k = max(
        1,
        min(
            int(top_k),
            available_labels,
        ),
    )

    top_probabilities, top_indices = (
        torch.topk(
            probabilities,
            k=top_k,
        )
    )

    top_predictions = []

    for probability, label_id in zip(
        top_probabilities.tolist(),
        top_indices.tolist(),
    ):
        label = _get_label(
            model.config.id2label,
            int(label_id),
        )

        top_predictions.append(
            {
                "intent": label,
                "confidence": float(
                    probability
                ),
            }
        )

    main_prediction = (
        top_predictions[0]
    )

    confidence = float(
        main_prediction["confidence"]
    )

    return {
        "text": clean_text,
        "intent":
            main_prediction["intent"],
        "confidence": confidence,
        "top_predictions":
            top_predictions,
        "low_confidence":
            confidence
            < LOW_CONFIDENCE_THRESHOLD,
        "device": str(device),
    }

INTENT_KEYWORD_RULES = [
    {
        "intent": "transfer_pending",
        "keywords": (
            "pending",
            "tertunda",
            "masih diproses",
            "belum selesai diproses",
        ),
    },
    {
        "intent": "transfer_failed",
        "keywords": (
            "transfer gagal",
            "gagal transfer",
            "gagal diproses",
            "transfer tidak berhasil",
        ),
    },
    {
        "intent": "suspicious_transaction",
        "keywords": (
            "tidak saya kenali",
            "tidak mengenali",
            "tidak dikenali",
            "tidak dikenal",
            "transaksi asing",
            "transaksi mencurigakan",
        ),
    },
    {
        "intent": "kyc_failed",
        "keywords": (
            "kyc gagal",
            "verifikasi identitas gagal",
            "verifikasi gagal",
            "identitas ditolak",
        ),
    },
    {
        "intent": "update_identity",
        "keywords": (
            "ubah identitas",
            "perbarui identitas",
            "update identitas",
            "perubahan identitas",
        ),
    },
    {
        "intent": "change_pin",
        "keywords": (
            "ganti pin",
            "ubah pin",
            "mengganti pin",
        ),
    },
    {
        "intent": "lost_card",
        "keywords": (
            "kartu hilang",
            "kehilangan kartu",
            "kartu dicuri",
        ),
    },
    {
        "intent": "account_blocked",
        "keywords": (
            "akun diblokir",
            "rekening diblokir",
            "akun terkunci",
        ),
    },
    {
        "intent": "balance_not_updated",
        "keywords": (
            "saldo belum masuk",
            "saldo belum berubah",
            "saldo tidak diperbarui",
            "saldo tidak bertambah",
        ),
    },
]


def resolve_intent(
    text: str,
    model_result: dict[str, Any] | None = None,
    confidence_threshold: float = 0.60,
) -> dict[str, Any]:
    """
    Menggabungkan hasil IndoBERT dengan aturan
    kata kunci untuk pesan yang sangat eksplisit.

    Aturan hanya menggantikan prediksi model apabila:
    1. confidence model rendah, atau
    2. model menandai low_confidence.
    """

    clean_text = str(text).strip()

    if not clean_text:
        raise ValueError(
            "Teks intent tidak boleh kosong."
        )

    if model_result is None:
        model_result = predict_intent(
            clean_text
        )

    resolved_result = dict(
        model_result
    )

    model_intent = str(
        model_result.get(
            "intent",
            "unknown",
        )
    )

    model_confidence = float(
        model_result.get(
            "confidence",
            0.0,
        )
    )

    model_low_confidence = bool(
        model_result.get(
            "low_confidence",
            False,
        )
    )

    lower_text = clean_text.lower()

    rule_intent: str | None = None
    matched_keyword: str | None = None

    for rule in INTENT_KEYWORD_RULES:
        for keyword in rule["keywords"]:
            if keyword in lower_text:
                rule_intent = str(
                    rule["intent"]
                )
                matched_keyword = keyword
                break

        if rule_intent is not None:
            break

    override_allowed = (
        model_low_confidence
        or model_confidence
        < confidence_threshold
    )

    override_applied = bool(
        rule_intent
        and rule_intent != model_intent
        and override_allowed
    )

    resolved_intent = (
        rule_intent
        if override_applied
        else model_intent
    )

    resolved_result.update(
        {
            "model_intent":
                model_intent,
            "model_confidence":
                model_confidence,
            "rule_intent":
                rule_intent,
            "matched_keyword":
                matched_keyword,
            "intent":
                resolved_intent,
            "override_applied":
                override_applied,
            "resolution_method": (
                "keyword_override"
                if override_applied
                else "indobert"
            ),
        }
    )

    return resolved_result