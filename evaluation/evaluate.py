import json
import re
import sys
from pathlib import Path

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag_graph import ABSTAIN, ask


QUESTIONS = Path(__file__).with_name("questions.json")


def normalize_for_evaluation(text: str) -> str:
    """
    Normalize text before evaluation.
    """

    text = str(text).lower()

    # Normalize dash variants
    text = text.replace("–", "-")  # en dash
    text = text.replace("—", "-")  # em dash
    text = text.replace("−", "-")  # mathematical minus

    # Normalize non-breaking spaces
    text = text.replace("\u00a0", " ")

    # Collapse repeated whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def main():
    cases = json.loads(
        QUESTIONS.read_text(encoding="utf-8")
    )

    passed = 0

    for case in cases:
        question = case["question"]

        result = ask(question)

        answer = result["answer"]

        normalized_answer = normalize_for_evaluation(
            answer
        )

        abstained = answer.strip() == ABSTAIN

        required_terms = [
            normalize_for_evaluation(term)
            for term in case["must_contain"]
        ]

        if case["must_not_abstain"]:
            ok = (
                not abstained
                and all(
                    term in normalized_answer
                    for term in required_terms
                )
            )
        else:
            ok = abstained

        passed += int(ok)

        print(
            f"[{'PASS' if ok else 'FAIL'}] "
            f"{question}"
        )

        print(
            f"      {answer}"
        )

        print(
            "      citations: "
            f"{[c['label'] for c in result['citations']]}"
        )

    print(
        f"\nScore: {passed}/{len(cases)}"
    )


if __name__ == "__main__":
    main()