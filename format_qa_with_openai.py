import os
import json
import time
import argparse
from typing import List, Dict, Any

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI


def load_api_key() -> None:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment. Create a .env with OPENAI_API_KEY=<key>.")


def build_messages(batch: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    system_instructions = (
        "You transform terse agricultural Q/A fragments into natural, well-formed farmer questions and advisor answers. "
        "Keep the original meaning and agronomic facts. Maintain the original language when possible; if unclear, default to clear English. "
        "For questions: write as a polite, direct question from a farmer. For answers: write as a concise, helpful advisor reply. "
        "Do not add new facts. Expand abbreviations (e.g., 'qty' -> 'quantity') when obvious. Normalize capitalization and punctuation. "
        "Return JSON with a top-level key 'items' containing a list of objects {id, question, answer}, where question and answer are the rewritten texts."
    )

    examples = [
        {
            "id": "ex1",
            "question": "asking for the fertiser dose of french bean.",
            "answer": "urea 30kg per acre, SSP 125kg per acre",
        },
        {
            "id": "ex2",
            "question": "powdery mildew control in okra",
            "answer": "use sulphur",
        },
    ]

    example_json = {
        "items": [
            {
                "id": "ex1",
                "question": "What is the recommended fertilizer dose for French bean?",
                "answer": "Apply approximately 30 kg urea and 125 kg single super phosphate (SSP) per acre, adjusting to soil test results.",
            },
            {
                "id": "ex2",
                "question": "How can I control powdery mildew in okra?",
                "answer": "Apply a wettable sulphur formulation as per label directions, ensure good coverage, and rotate modes of action.",
            },
        ]
    }

    content_lines = [
        "You will receive a JSON array of Q/A items with keys: id, question, answer.",
        "Rewrite each item as natural language Q and A without changing meaning.",
        "Respond ONLY with a single JSON object in this exact schema: {\"items\":[{\"id\":str,\"question\":str,\"answer\":str},...]}",
        "Example input:",
        json.dumps(examples, ensure_ascii=False),
        "Example output:",
        json.dumps(example_json, ensure_ascii=False),
        "Input:",
        json.dumps(batch, ensure_ascii=False),
    ]

    return [
        {"role": "system", "content": system_instructions},
        {"role": "user", "content": "\n\n".join(content_lines)},
    ]


def call_openai(client: OpenAI, messages: List[Dict[str, str]], max_retries: int = 5, timeout_s: float = 30.0) -> Dict[str, Any]:
    attempt = 0
    while True:
        try:
            response = client.chat.completions.create(
                model="gpt-5",
                messages=messages,
                temperature=1,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as exc:  # noqa: BLE001
            attempt += 1
            if attempt >= max_retries:
                raise
            sleep_s = min(2 ** attempt, 30)
            time.sleep(sleep_s)


def process_dataframe(df: pd.DataFrame, batch_size: int, start: int | None, limit: int | None) -> pd.DataFrame:
    df = df.copy()
    total_rows = len(df)

    begin = 0 if start is None else max(0, start)
    end = total_rows if limit is None else min(total_rows, begin + max(0, limit))

    client = OpenAI()

    formatted_questions: List[str] = [None] * total_rows  # type: ignore[assignment]
    formatted_answers: List[str] = [None] * total_rows  # type: ignore[assignment]

    cursor = begin
    while cursor < end:
        slice_end = min(cursor + batch_size, end)
        batch_records: List[Dict[str, Any]] = []
        for row_index in range(cursor, slice_end):
            row = df.iloc[row_index]
            raw_q = None if pd.isna(row["questions"]) else str(row["questions"])  # type: ignore[index]
            raw_a = None if pd.isna(row["answers"]) else str(row["answers"])  # type: ignore[index]
            if not raw_q or not raw_a:
                continue
            batch_records.append({"id": str(row_index), "question": raw_q, "answer": raw_a})

        if not batch_records:
            cursor = slice_end
            continue

        messages = build_messages(batch_records)
        result = call_openai(client, messages)
        items = result.get("items", [])
        mapped = {str(item.get("id")): item for item in items}

        for row_index in range(cursor, slice_end):
            key = str(row_index)
            item = mapped.get(key)
            if item is None:
                continue
            q_text = item.get("question")
            a_text = item.get("answer")
            if isinstance(q_text, str) and q_text.strip():
                formatted_questions[row_index] = q_text.strip()
            if isinstance(a_text, str) and a_text.strip():
                formatted_answers[row_index] = a_text.strip()

        processed_now = slice_end - begin
        processed_total = (slice_end - begin)
        print(f"Processed rows {cursor}-{slice_end - 1} ({processed_now} in this batch)")
        cursor = slice_end

    df["question_formatted"] = [
        (formatted_questions[i] if formatted_questions[i] is not None else df.iloc[i]["questions"])  # type: ignore[index]
        for i in range(total_rows)
    ]
    df["answer_formatted"] = [
        (formatted_answers[i] if formatted_answers[i] is not None else df.iloc[i]["answers"])  # type: ignore[index]
        for i in range(total_rows)
    ]
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite farmer Q/A to well-formed questions and answers using OpenAI.")
    parser.add_argument("--input", type=str, default="questionsv4_cleaned.csv", help="Input CSV path (expects columns: questions, answers)")
    parser.add_argument("--output", type=str, default="questionsv4_formatted.csv", help="Output CSV path")
    parser.add_argument("--batch-size", type=int, default=20, help="Number of rows per API request")
    parser.add_argument("--start", type=int, default=None, help="Start row index (inclusive)")
    parser.add_argument("--limit", type=int, default=None, help="Max number of rows to process")
    args = parser.parse_args()

    load_api_key()

    df = pd.read_csv(args.input, dtype=str)
    if not {"questions", "answers"}.issubset({c.strip().lower() for c in df.columns}):
        raise RuntimeError("Input CSV must have 'questions' and 'answers' columns.")

    df.columns = [c.strip().lower() for c in df.columns]
    df_result = process_dataframe(df, batch_size=args.batch_size, start=args.start, limit=args.limit)
    df_result.to_csv(args.output, index=False)
    print(f"Saved formatted CSV to {args.output}")


if __name__ == "__main__":
    main()


