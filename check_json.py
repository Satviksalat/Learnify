import json
import os

files = [
    'backend/data/exam_questions.json',
    'backend/data/programs.json',
    'backend/data/quizzes.json'
]

print("Checking files...")
for f in files:
    try:
        if os.path.exists(f):
            with open(f, 'r') as fp:
                json.load(fp)
            print(f"{f}: VALID")
        else:
            print(f"{f}: NOT FOUND")
    except json.JSONDecodeError as e:
        print(f"{f}: INVALID JSON - {e}")
    except Exception as e:
        print(f"{f}: ERROR - {e}")
