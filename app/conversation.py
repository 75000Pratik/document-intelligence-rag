import json
from pathlib import Path

HISTORY_FILE = Path("data/conversation_history.json")


def load_history():
    if not HISTORY_FILE.exists():
        return {}

    with open(
        HISTORY_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def save_history(history):
    HISTORY_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        HISTORY_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            history,
            file,
            indent=4
        )


conversation_history = load_history()


def add_message(conversation_id, question, answer):
    if conversation_id not in conversation_history:
        conversation_history[conversation_id] = []

    conversation_history[conversation_id].append(
        {
            "question": question,
            "answer": answer
        }
    )

    save_history(
        conversation_history
    )


def get_history(conversation_id):
    return conversation_history.get(
        conversation_id,
        []
    )
