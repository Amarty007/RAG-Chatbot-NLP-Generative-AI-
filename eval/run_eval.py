"""Run a small reference-question evaluation against the indexed PDF."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rag import answer, grade_answer  # noqa: E402


def load_questions(path: Path) -> list[dict[str, Any]]:
    """Load and validate the evaluation question list."""
    with path.open(encoding="utf-8") as questions_file:
        questions = json.load(questions_file)
    if not isinstance(questions, list):
        raise ValueError("The evaluation file must contain a JSON list.")
    for index, item in enumerate(questions):
        if not all(key in item for key in ("question", "reference_answer", "page")):
            raise ValueError(f"Question {index + 1} needs question, reference_answer, and page.")
    return questions


def main() -> None:
    """Run answer generation and LLM grading for each reference question."""
    parser = argparse.ArgumentParser(description="Evaluate the PDF RAG chatbot.")
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path(__file__).with_name("questions.json"),
        help="JSON file containing question, reference_answer, and one-based page fields",
    )
    args = parser.parse_args()
    questions = load_questions(args.questions)
    if not questions:
        print(f"No evaluation questions found in {args.questions}.")
        print("Add questions with question, reference_answer, and one-based page fields.")
        return

    total_score = 0.0
    correct_citations = 0
    for index, item in enumerate(questions, start=1):
        model_answer, sources = answer(item["question"])
        grade = grade_answer(
            item["question"],
            item["reference_answer"],
            model_answer,
            int(item["page"]),
        )
        total_score += grade["score"]
        correct_citations += int(grade["citation_correct"])
        pages = sorted({source["page"] for source in sources})
        print(f"\nQuestion {index}: {item['question']}")
        print(f"Answer: {model_answer}")
        print(f"Retrieved pages: {pages}")
        print(
            f"Score: {grade['score']:.1f} | "
            f"Citation correct: {'yes' if grade['citation_correct'] else 'no'}"
        )
        if grade.get("reason"):
            print(f"Reason: {grade['reason']}")

    count = len(questions)
    print(f"\nAnswer accuracy (mean rubric score): {total_score / count:.1%}")
    print(f"Citation accuracy: {correct_citations / count:.1%}")


if __name__ == "__main__":
    main()
