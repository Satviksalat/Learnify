import json
try:
    with open(r'd:/satvik/ANTGVTPROJECT/PYProj/TutorialPlatform/backend/data/tutorials.json', 'r') as f:
        json.load(f)
    print("JSON is VALID")
except Exception as e:
    print(f"JSON INVALID: {e}")
