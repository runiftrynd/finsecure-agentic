from __future__ import annotations

import re
from typing import Any

from rapidfuzz import fuzz

from model_loader import (
    load_kyc_dataframe,
)


REQUIRED_KYC_FIELDS = [
    "form_name",
    "document_name",
    "form_nik",
    "document_nik",
    "form_birth_date",
    "document_birth_date",
    "document_status",
]


def validate_kyc_record(
    record: dict[str, Any],
) -> dict[str, Any]:
    """
    Memvalidasi satu data KYC menggunakan
    aturan yang sama dengan notebook Colab.
    """

    missing_fields = [
        field
        for field in REQUIRED_KYC_FIELDS
        if not str(
            record.get(field, "")
        ).strip()
    ]

    issues = []

    if missing_fields:
        issues.append(
            "missing_required_fields"
        )

    form_nik = str(
        record.get(
            "form_nik",
            "",
        )
    ).strip()

    document_nik = str(
        record.get(
            "document_nik",
            "",
        )
    ).strip()

    nik_format_valid = bool(
        re.fullmatch(
            r"\d{16}",
            form_nik,
        )
    ) and bool(
        re.fullmatch(
            r"\d{16}",
            document_nik,
        )
    )

    if not nik_format_valid:
        issues.append(
            "invalid_nik_format"
        )

    nik_match = (
        form_nik == document_nik
        and bool(form_nik)
    )

    if not nik_match:
        issues.append(
            "nik_mismatch"
        )

    form_birth_date = str(
        record.get(
            "form_birth_date",
            "",
        )
    ).strip()

    document_birth_date = str(
        record.get(
            "document_birth_date",
            "",
        )
    ).strip()

    birth_date_match = (
        form_birth_date
        == document_birth_date
        and bool(form_birth_date)
    )

    if not birth_date_match:
        issues.append(
            "birth_date_mismatch"
        )

    document_status = str(
        record.get(
            "document_status",
            "",
        )
    ).strip().lower()

    if document_status != "active":
        issues.append(
            "document_not_active"
        )

    form_name = str(
        record.get(
            "form_name",
            "",
        )
    ).strip().lower()

    document_name = str(
        record.get(
            "document_name",
            "",
        )
    ).strip().lower()

    name_similarity = float(
        fuzz.ratio(
            form_name,
            document_name,
        )
    )

    if name_similarity < 75:
        issues.append(
            "significant_name_mismatch"
        )

    elif name_similarity < 95:
        issues.append(
            "minor_name_variation"
        )

    critical_issues = {
        "missing_required_fields",
        "invalid_nik_format",
        "nik_mismatch",
        "birth_date_mismatch",
        "document_not_active",
        "significant_name_mismatch",
    }

    has_critical_issue = any(
        issue in critical_issues
        for issue in issues
    )

    if has_critical_issue:
        status = "reject"

    elif name_similarity < 95:
        status = "review"

    else:
        status = "approve"

    return {
        "application_id":
            record.get(
                "application_id"
            ),
        "status":
            status,
        "name_similarity":
            name_similarity,
        "nik_format_valid":
            nik_format_valid,
        "nik_match":
            nik_match,
        "birth_date_match":
            birth_date_match,
        "document_status":
            document_status,
        "missing_fields":
            missing_fields,
        "issues":
            issues,
        "human_review_required":
            status in {
                "review",
                "reject",
            },
    }


def get_kyc_record(
    application_id: str,
) -> dict[str, Any] | None:
    """
    Mengambil satu record KYC berdasarkan ID.
    """

    clean_application_id = str(
        application_id
    ).strip().upper()

    if not clean_application_id:
        raise ValueError(
            "Application ID tidak boleh kosong."
        )

    dataframe = load_kyc_dataframe()

    selected_rows = dataframe[
        dataframe["application_id"]
        == clean_application_id
    ]

    if selected_rows.empty:
        return None

    return selected_rows.iloc[
        0
    ].to_dict()


def validate_kyc_application(
    application_id: str,
) -> dict[str, Any]:
    """
    Mengambil dan memvalidasi data KYC.
    """

    clean_application_id = str(
        application_id
    ).strip().upper()

    record = get_kyc_record(
        clean_application_id
    )

    if record is None:
        return {
            "status": "not_found",
            "application_id":
                clean_application_id,
            "message":
                "Application ID tidak ditemukan.",
            "human_review_required":
                False,
        }

    return validate_kyc_record(
        record
    )