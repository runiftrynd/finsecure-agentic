from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import chromadb
import pandas as pd
import torch
import xgboost as xgb
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from sentence_transformers import (
    SentenceTransformer,
)


# ============================================================
# KONFIGURASI DASAR
# ============================================================

os.environ.setdefault(
    "TOKENIZERS_PARALLELISM",
    "false",
)

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

VECTORSTORE_DIR = (
    PROJECT_ROOT
    / "vectorstore"
)

EMBEDDING_MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

VECTORSTORE_CONFIG = {
    "customer_service": {
        "directory":
            VECTORSTORE_DIR
            / "customer_service",
        "collection_name":
            "finsecure_customer_service",
    },
    "fraud_risk": {
        "directory":
            VECTORSTORE_DIR
            / "fraud_risk",
        "collection_name":
            "finsecure_fraud_risk",
    },
    "kyc_compliance": {
        "directory":
            VECTORSTORE_DIR
            / "kyc_compliance",
        "collection_name":
            "finsecure_kyc_compliance",
    },
}

INTENT_MODEL_DIR = (
    MODEL_DIR
    / "intent_indobert"
)

FRAUD_MODEL_PATH = (
    MODEL_DIR
    / "fraud_model.json"
)

FRAUD_METADATA_PATH = (
    MODEL_DIR
    / "fraud_metadata.json"
)

KYC_DATA_PATH = (
    DATA_DIR
    / "kyc"
    / "kyc_dataset.csv"
)

TRANSACTION_DATA_PATH = (
    DATA_DIR
    / "transactions_demo.csv"
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# VALIDASI PATH
# ============================================================

def require_path(
    path: Path,
    description: str,
) -> Path:
    """
    Memastikan file atau folder tersedia.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"{description} tidak ditemukan: {path}"
        )

    return path


# ============================================================
# LOADER INTENT INDOBERT
# ============================================================

@lru_cache(maxsize=1)
def load_intent_resources() -> tuple[
    Any,
    Any,
    torch.device,
]:
    """
    Memuat tokenizer dan model IndoBERT satu kali.
    """

    require_path(
        INTENT_MODEL_DIR,
        "Folder model intent IndoBERT",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        str(INTENT_MODEL_DIR),
        local_files_only=True,
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            str(INTENT_MODEL_DIR),
            local_files_only=True,
        )
    )

    model.to(DEVICE)
    model.eval()

    return tokenizer, model, DEVICE


# ============================================================
# LOADER MODEL FRAUD
# ============================================================

@lru_cache(maxsize=1)
def load_fraud_resources() -> tuple[
    Any,
    list[str],
    float,
    dict[str, Any],
]:
    """
    Memuat model XGBoost dan metadata fraud satu kali.
    """

    require_path(
        FRAUD_MODEL_PATH,
        "Model fraud",
    )

    require_path(
        FRAUD_METADATA_PATH,
        "Metadata model fraud",
    )

    with open(
        FRAUD_METADATA_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    fraud_features = metadata.get(
        "features",
        [],
    )

    fraud_threshold = metadata.get(
        "threshold",
    )

    model_type = metadata.get(
        "model_type",
        "XGBClassifier",
    )

    if not fraud_features:
        raise ValueError(
            "Daftar fitur fraud tidak ditemukan "
            "dalam fraud_metadata.json."
        )

    if fraud_threshold is None:
        raise ValueError(
            "Threshold fraud tidak ditemukan "
            "dalam fraud_metadata.json."
        )

    if model_type == "Booster":
        fraud_model = xgb.Booster(
            params={
                "device": "cpu",
            }
        )

        fraud_model.load_model(
            str(FRAUD_MODEL_PATH)
        )

    else:
        fraud_model = xgb.XGBClassifier(
            device="cpu",
            n_jobs=-1,
        )

        fraud_model.load_model(
            str(FRAUD_MODEL_PATH)
        )

    return (
        fraud_model,
        list(fraud_features),
        float(fraud_threshold),
        metadata,
    )


# ============================================================
# LOADER DATASET KYC
# ============================================================

@lru_cache(maxsize=1)
def load_kyc_dataframe() -> pd.DataFrame:
    """
    Memuat dan memvalidasi dataset KYC.
    """

    require_path(
        KYC_DATA_PATH,
        "Dataset KYC",
    )

    dataframe = pd.read_csv(
        KYC_DATA_PATH,
        dtype={
            "application_id": str,
            "form_name": str,
            "document_name": str,
            "form_nik": str,
            "document_nik": str,
            "form_birth_date": str,
            "document_birth_date": str,
            "document_status": str,
            "expected_status": str,
        },
        keep_default_na=False,
    )

    required_columns = {
        "application_id",
        "form_name",
        "document_name",
        "form_nik",
        "document_nik",
        "form_birth_date",
        "document_birth_date",
        "document_status",
        "expected_status",
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Kolom dataset KYC tidak lengkap: "
            f"{sorted(missing_columns)}"
        )

    dataframe = dataframe.copy()

    for column in required_columns:
        dataframe[column] = (
            dataframe[column]
            .astype(str)
            .str.strip()
        )

    dataframe["application_id"] = (
        dataframe["application_id"]
        .str.upper()
    )

    if dataframe.empty:
        raise ValueError(
            "Dataset KYC kosong."
        )

    if (
        dataframe["application_id"]
        .duplicated()
        .any()
    ):
        raise ValueError(
            "Dataset KYC memiliki "
            "application_id duplikat."
        )

    return dataframe

# ============================================================
# LOADER DATASET TRANSAKSI
# ============================================================
@lru_cache(maxsize=1)
def load_transaction_dataframe() -> pd.DataFrame:
    """
    Memuat database transaksi demo.
    """

    require_path(
        TRANSACTION_DATA_PATH,
        "Dataset transaksi demo",
    )

    dataframe = pd.read_csv(
        TRANSACTION_DATA_PATH,
        dtype={
            "transaction_id": str,
            "customer_id": str,
            "transaction_status": str,
            "description": str,
        },
        keep_default_na=False,
    )

    (
        _,
        fraud_features,
        _,
        _,
    ) = load_fraud_resources()

    required_columns = {
        "transaction_id",
        "customer_id",
        "transaction_status",
        "description",
        *fraud_features,
    }

    missing_columns = (
        required_columns
        - set(dataframe.columns)
    )

    if missing_columns:
        raise ValueError(
            "Kolom dataset transaksi tidak lengkap: "
            f"{sorted(missing_columns)}"
        )

    dataframe = dataframe.copy()

    dataframe["transaction_id"] = (
        dataframe["transaction_id"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    dataframe["customer_id"] = (
        dataframe["customer_id"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    dataframe["transaction_status"] = (
        dataframe["transaction_status"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    dataframe["description"] = (
        dataframe["description"]
        .astype(str)
        .str.strip()
    )

    for feature in fraud_features:
        dataframe[feature] = pd.to_numeric(
            dataframe[feature],
            errors="raise",
        )

    if dataframe.empty:
        raise ValueError(
            "Dataset transaksi demo kosong."
        )

    if (
        dataframe["transaction_id"]
        .duplicated()
        .any()
    ):
        raise ValueError(
            "Dataset transaksi memiliki "
            "transaction_id duplikat."
        )

    return dataframe

# ============================================================
# LOADER EMBEDDING MODEL
# ============================================================

@lru_cache(maxsize=1)
def load_embedding_model() -> SentenceTransformer:
    """
    Memuat model embedding yang sama dengan
    model yang digunakan saat pembuatan vectorstore.
    """

    embedding_device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    model = SentenceTransformer(
        EMBEDDING_MODEL_NAME,
        device=embedding_device,
    )

    return model


# ============================================================
# LOADER CHROMADB CLIENT
# ============================================================

@lru_cache(maxsize=3)
def load_vector_client(
    domain: str,
) -> Any:
    """
    Memuat persistent ChromaDB berdasarkan domain.
    """

    clean_domain = str(
        domain
    ).strip().lower()

    if clean_domain not in VECTORSTORE_CONFIG:
        raise ValueError(
            "Domain vectorstore tidak tersedia: "
            f"{clean_domain}. "
            "Domain yang valid adalah "
            "customer_service, fraud_risk, "
            "dan kyc_compliance."
        )

    configuration = VECTORSTORE_CONFIG[
        clean_domain
    ]

    vectorstore_directory = configuration[
        "directory"
    ]

    require_path(
        vectorstore_directory,
        f"Vectorstore {clean_domain}",
    )

    return chromadb.PersistentClient(
        path=str(vectorstore_directory)
    )


# ============================================================
# LOADER CHROMADB COLLECTION
# ============================================================

@lru_cache(maxsize=3)
def load_vector_collection(
    domain: str,
) -> Any:
    """
    Memuat collection ChromaDB berdasarkan domain.
    """

    clean_domain = str(
        domain
    ).strip().lower()

    if clean_domain not in VECTORSTORE_CONFIG:
        raise ValueError(
            "Domain vectorstore tidak tersedia: "
            f"{clean_domain}"
        )

    client = load_vector_client(
        clean_domain
    )

    collection_name = (
        VECTORSTORE_CONFIG[
            clean_domain
        ]["collection_name"]
    )

    try:
        collection = client.get_collection(
            name=collection_name
        )

    except Exception as error:
        raise RuntimeError(
            "Collection ChromaDB tidak ditemukan: "
            f"{collection_name}"
        ) from error

    if collection.count() == 0:
        raise ValueError(
            "Collection ChromaDB kosong: "
            f"{collection_name}"
        )

    return collection
# ============================================================
# RINGKASAN RESOURCE
# ============================================================

def get_resource_summary() -> dict[str, Any]:
    """
    Menampilkan ringkasan resource aplikasi.
    """

    _, intent_model, intent_device = (
        load_intent_resources()
    )

    (
        _,
        fraud_features,
        fraud_threshold,
        fraud_metadata,
    ) = load_fraud_resources()

    kyc_dataframe = load_kyc_dataframe()
    transaction_dataframe = (load_transaction_dataframe())

    return {
        "intent": {
            "device": str(intent_device),
            "num_labels":
                intent_model.config.num_labels,
            "labels":
                intent_model.config.id2label,
        },
        "fraud": {
            "model_type":
                fraud_metadata.get(
                    "model_type"
                ),
            "feature_count":
                len(fraud_features),
            "threshold":
                fraud_threshold,
            "xgboost_version":
                fraud_metadata.get(
                    "xgboost_version"
                ),
        },
        "kyc": {
            "row_count":
                len(kyc_dataframe),
            "column_count":
                len(
                    kyc_dataframe.columns
                ),
        },
        "transactions": {
            "row_count":
                len(transaction_dataframe),
            "column_count":
                len(transaction_dataframe.columns),
        },
    }