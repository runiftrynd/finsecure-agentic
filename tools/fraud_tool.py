from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from model_loader import load_fraud_resources


FEATURE_DESCRIPTIONS = {
    "TX_AMOUNT": "Nominal transaksi",
    "TX_DURING_WEEKEND": "Transaksi dilakukan saat akhir pekan",
    "TX_DURING_NIGHT": "Transaksi dilakukan pada malam hari",
    "CUSTOMER_ID_NB_TX_1DAY_WINDOW":
        "Jumlah transaksi nasabah dalam satu hari",
    "CUSTOMER_ID_NB_TX_7DAY_WINDOW":
        "Jumlah transaksi nasabah dalam tujuh hari",
    "CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW":
        "Rata-rata nominal nasabah selama tujuh hari",
    "CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW":
        "Rata-rata nominal nasabah selama tiga puluh hari",
    "TERMINAL_ID_NB_TX_7DAY_WINDOW":
        "Jumlah transaksi terminal selama tujuh hari",
    "TERMINAL_ID_RISK_7DAY_WINDOW":
        "Tingkat risiko terminal selama tujuh hari",
    "TERMINAL_ID_RISK_30DAY_WINDOW":
        "Tingkat risiko terminal selama tiga puluh hari",
}


def _prepare_feature_dataframe(
    input_features: Mapping[str, Any],
    required_features: list[str],
) -> pd.DataFrame:
    """
    Memeriksa, membersihkan, dan mengurutkan fitur fraud.
    """

    missing_features = [
        feature
        for feature in required_features
        if feature not in input_features
    ]

    if missing_features:
        raise ValueError(
            "Fitur fraud belum lengkap: "
            f"{missing_features}"
        )

    clean_features: dict[str, float] = {}

    for feature in required_features:
        value = input_features.get(feature)

        if value is None or str(value).strip() == "":
            raise ValueError(
                f"Nilai fitur {feature} tidak boleh kosong."
            )

        try:
            clean_features[feature] = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Nilai fitur {feature} harus berupa angka."
            ) from error

    return pd.DataFrame(
        [clean_features],
        columns=required_features,
    )


def _predict_probability(
    model: Any,
    dataframe: pd.DataFrame,
    feature_names: list[str],
) -> float:
    """
    Menghasilkan probabilitas transaksi fraud.
    """

    if isinstance(model, xgb.Booster):
        data_matrix = xgb.DMatrix(
            dataframe,
            feature_names=feature_names,
        )

        prediction = model.predict(data_matrix)

        return float(prediction[0])

    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "Model fraud tidak memiliki fungsi predict_proba()."
        )

    prediction = model.predict_proba(dataframe)

    if prediction.ndim != 2 or prediction.shape[1] < 2:
        raise ValueError(
            "Output predict_proba model fraud tidak valid."
        )

    return float(prediction[0][1])


def _get_booster(model: Any) -> xgb.Booster:
    """
    Mengambil objek Booster dari model XGBoost.
    """

    if isinstance(model, xgb.Booster):
        return model

    if hasattr(model, "get_booster"):
        return model.get_booster()

    raise TypeError(
        "Model tidak dapat dikonversi menjadi XGBoost Booster."
    )


def explain_fraud_prediction(
    model: Any,
    dataframe: pd.DataFrame,
    feature_names: list[str],
    maximum_factors: int = 4,
) -> list[dict[str, Any]]:
    """
    Mengambil fitur yang memberikan kontribusi positif
    terhadap prediksi risiko fraud.
    """

    try:
        booster = _get_booster(model)

        data_matrix = xgb.DMatrix(
            dataframe,
            feature_names=feature_names,
        )

        contributions = booster.predict(
            data_matrix,
            pred_contribs=True,
        )[0]

        # Elemen terakhir adalah nilai bias.
        feature_contributions = contributions[:-1]

        sorted_indices = np.argsort(
            feature_contributions
        )[::-1]

        risk_factors: list[dict[str, Any]] = []

        for feature_index in sorted_indices:
            contribution = float(
                feature_contributions[feature_index]
            )

            if contribution <= 0:
                continue

            feature_name = feature_names[
                int(feature_index)
            ]

            risk_factors.append(
                {
                    "feature": feature_name,
                    "description":
                        FEATURE_DESCRIPTIONS.get(
                            feature_name,
                            feature_name,
                        ),
                    "value": float(
                        dataframe.iloc[0][feature_name]
                    ),
                    "contribution": contribution,
                }
            )

            if len(risk_factors) >= maximum_factors:
                break

        return risk_factors

    except Exception as error:
        # Kegagalan explainability tidak menggagalkan prediksi.
        return [
            {
                "feature": "explanation_unavailable",
                "description":
                    "Kontribusi fitur tidak dapat dihitung.",
                "value": None,
                "contribution": None,
                "error": str(error),
            }
        ]


def analyze_fraud(
    input_features: Mapping[str, Any],
    transaction_id: str | None = None,
) -> dict[str, Any]:
    """
    Menjalankan prediksi fraud untuk satu transaksi.
    """

    (
        fraud_model,
        fraud_features,
        fraud_threshold,
        fraud_metadata,
    ) = load_fraud_resources()

    feature_dataframe = _prepare_feature_dataframe(
        input_features=input_features,
        required_features=fraud_features,
    )

    fraud_probability = _predict_probability(
        model=fraud_model,
        dataframe=feature_dataframe,
        feature_names=fraud_features,
    )

    model_prediction = int(
        fraud_probability >= fraud_threshold
    )

    if fraud_probability >= 0.80:
        risk_level = "high"
        recommendation = "manual_review"
        human_review_required = True

    elif fraud_probability >= 0.50:
        risk_level = "medium"
        recommendation = "additional_verification"
        human_review_required = True

    else:
        risk_level = "low"
        recommendation = "continue_monitoring"
        human_review_required = False

    risk_factors = explain_fraud_prediction(
        model=fraud_model,
        dataframe=feature_dataframe,
        feature_names=fraud_features,
    )

    return {
        "status": "success",
        "transaction_id": transaction_id,
        "fraud_probability": fraud_probability,
        "fraud_probability_percent":
            round(fraud_probability * 100, 4),
        "model_threshold": fraud_threshold,
        "model_prediction": model_prediction,
        "is_fraud": bool(model_prediction),
        "risk_level": risk_level,
        "risk_factors": risk_factors,
        "recommendation": recommendation,
        "human_review_required":
            human_review_required,
        "feature_count": len(fraud_features),
        "model_type": fraud_metadata.get(
            "model_type",
            type(fraud_model).__name__,
        ),
    }