from __future__ import annotations

import os
import time
import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any

from google import genai


# ============================================================
# KONFIGURASI
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

STREAMLIT_SECRETS_PATH = (
    PROJECT_ROOT
    / ".streamlit"
    / "secrets.toml"
)

DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"


# ============================================================
# MEMBACA KONFIGURASI
# ============================================================

def _read_streamlit_secrets() -> dict[str, Any]:
    """
    Membaca secrets.toml untuk pengujian lokal.

    File ini tidak boleh dimasukkan ke GitHub.
    """

    if not STREAMLIT_SECRETS_PATH.exists():
        return {}

    try:
        with open(
            STREAMLIT_SECRETS_PATH,
            "rb",
        ) as file:
            return tomllib.load(file)

    except Exception as error:
        raise RuntimeError(
            "File .streamlit/secrets.toml "
            "tidak dapat dibaca."
        ) from error


def get_gemini_settings() -> tuple[str, str]:
    """
    Mengambil API key dan model Gemini.

    Prioritas:
    1. Environment variable
    2. .streamlit/secrets.toml
    """

    local_secrets = _read_streamlit_secrets()

    api_key = (
        os.getenv("GEMINI_API_KEY")
        or local_secrets.get("GEMINI_API_KEY")
    )

    model_name = (
        os.getenv("GEMINI_MODEL")
        or local_secrets.get("GEMINI_MODEL")
        or DEFAULT_GEMINI_MODEL
    )

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY belum dikonfigurasi. "
            "Masukkan API key ke "
            ".streamlit/secrets.toml."
        )

    return str(api_key), str(model_name)


# ============================================================
# GEMINI CLIENT
# ============================================================

@lru_cache(maxsize=1)
def get_gemini_client() -> genai.Client:
    """
    Membuat Gemini client satu kali.
    """

    api_key, _ = get_gemini_settings()

    return genai.Client(
        api_key=api_key
    )


# ============================================================
# FALLBACK
# ============================================================

def create_fallback_response(
    reason: str | None = None,
) -> str:
    """
    Jawaban aman ketika Gemini tidak tersedia.

    Detail error disimpan pada field error,
    tetapi tidak ditampilkan kepada pengguna.
    """

    return (
        "Layanan penyusunan jawaban sedang tidak tersedia. "
        "Hasil pemeriksaan lokal tetap berhasil diproses "
        "dan dapat digunakan sebagai dasar tindak lanjut."
    )


# ============================================================
# GENERATE TEXT
# ============================================================

def generate_text(
    prompt: str,
    max_retries: int = 2,
) -> dict[str, Any]:
    """
    Mengirim prompt ke Gemini Interactions API.

    Mengembalikan fallback apabila API gagal.
    """

    clean_prompt = str(prompt).strip()

    if not clean_prompt:
        raise ValueError(
            "Prompt Gemini tidak boleh kosong."
        )

    api_key, model_name = get_gemini_settings()

    # Pastikan variabel digunakan, tanpa mencetak key.
    if not api_key:
        raise RuntimeError(
            "API key Gemini tidak tersedia."
        )

    client = get_gemini_client()

    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            interaction = client.interactions.create(
                model=model_name,
                input=clean_prompt,
            )

            response_text = str(
                interaction.output_text or ""
            ).strip()

            if not response_text:
                raise RuntimeError(
                    "Gemini mengembalikan jawaban kosong."
                )

            usage = getattr(
                interaction,
                "usage",
                None,
            )

            return {
                "status": "success",
                "provider": "gemini",
                "model": model_name,
                "response": response_text,
                "fallback_used": False,
                "usage": str(usage)
                if usage is not None
                else None,
                "error": None,
            }

        except Exception as error:
            last_error = error

            error_text = str(error).lower()

            retryable_error = any(
                keyword in error_text
                for keyword in [
                    "429",
                    "resource_exhausted",
                    "timeout",
                    "timed out",
                    "503",
                    "unavailable",
                    "temporarily",
                ]
            )

            if (
                retryable_error
                and attempt < max_retries
            ):
                wait_seconds = 2 ** attempt

                time.sleep(wait_seconds)
                continue

            break

    error_message = (
        str(last_error)
        if last_error is not None
        else "Kesalahan tidak diketahui."
    )

    return {
        "status": "fallback",
        "provider": "local_fallback",
        "model": model_name,
        "response": create_fallback_response(
            reason=error_message,
        ),
        "fallback_used": True,
        "usage": None,
        "error": error_message,
    }


# ============================================================
# STATUS CLIENT
# ============================================================

def get_llm_status() -> dict[str, Any]:
    """
    Memeriksa konfigurasi tanpa menampilkan API key.
    """

    try:
        _, model_name = get_gemini_settings()

        return {
            "configured": True,
            "provider": "gemini",
            "model": model_name,
            "secrets_file_exists":
                STREAMLIT_SECRETS_PATH.exists(),
        }

    except Exception as error:
        return {
            "configured": False,
            "provider": "gemini",
            "model": None,
            "secrets_file_exists":
                STREAMLIT_SECRETS_PATH.exists(),
            "error": str(error),
        }