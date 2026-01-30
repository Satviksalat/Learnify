
import json
import os

EXAM_FILE = os.path.join("backend", "data", "exam_questions.json")

with open(EXAM_FILE, 'r') as f:
    data = json.load(f)

print("--- Existing Units ---")
units = [u['unit'] for u in data]
for u in units:
    print(u)
