import os
import json
import datetime

FEEDBACK_DIR = os.path.join(os.path.dirname(__file__), '..', 'feedback')
FEEDBACK_FILE = os.path.join(FEEDBACK_DIR, 'feedback.jsonl')


def save_feedback(message, predicted_label, user_feedback):
    os.makedirs(FEEDBACK_DIR, exist_ok=True)
    entry = {
        'message': message,
        'predicted_label': predicted_label,
        'user_feedback': user_feedback,
        'timestamp': datetime.datetime.now().isoformat(),
    }
    with open(FEEDBACK_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry) + '\n')


def get_feedback_count():
    if not os.path.exists(FEEDBACK_FILE):
        return 0
    with open(FEEDBACK_FILE, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)


def get_all_feedback():
    if not os.path.exists(FEEDBACK_FILE):
        return []
    entries = []
    with open(FEEDBACK_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries
