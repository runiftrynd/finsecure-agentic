from __future__ import annotations

import json

from tools.intent_tool import (
    resolve_intent,
)


TEST_MESSAGES = [
    "Transfer saya masih pending.",
    "Transfer saya gagal diproses.",
    "Saya tidak mengenali transaksi ini.",
    "Verifikasi identitas saya gagal.",
    "Nama pada data identitas saya sedikit berbeda.",
    "Bagaimana cara mengganti PIN?",
]


def main() -> None:
    print("=" * 70)
    print("TEST HYBRID INTENT RESOLVER")
    print("=" * 70)

    for message in TEST_MESSAGES:
        result = resolve_intent(
            message
        )

        print("\nPesan:")
        print(message)

        print(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        )

    print("\n" + "=" * 70)
    print("TEST INTENT RESOLVER SELESAI")
    print("=" * 70)


if __name__ == "__main__":
    main()