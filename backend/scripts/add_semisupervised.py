import json
import os

JSON_PATH = r'd:/satvik/ANTGVTPROJECT/PYProj/TutorialPlatform/backend/data/tutorials.json'

NEW_ITEM = {
    "id": "ml-unit3-semisupervised",
    "title": "10. Semi-Supervised Learning",
    "technology": "Machine Learning with Python",
    "unit": "Unit 3: Unsupervised Learning",
    "definition": "Hybrid Approach",
    "description": "Combining labeled and unlabeled data.",
    "syntax": "Theory",
    "code_example": "# Train on Labeled -> Predict Unlabeled -> Retrain",
    "explanation": "Placeholder",
    "try_it_yourself": True,
    "key_points": [
        "Small Labeled Data",
        "Large Unlabeled Data",
        "Label Propagation"
    ]
}

def add_tutorial():
    if not os.path.exists(JSON_PATH):
        print("Error: tutorials.json not found")
        return

    with open(JSON_PATH, 'r') as f:
        data = json.load(f)

    # Check if already exists
    for t in data:
        if t['id'] == NEW_ITEM['id']:
            print("Item already exists.")
            return

    # Find the insertion point (After the last Unit 3 item)
    insert_index = -1
    for i, t in enumerate(data):
        if t.get('unit') == "Unit 3: Unsupervised Learning":
            insert_index = i
    
    # Insert after the last found item
    if insert_index != -1:
        data.insert(insert_index + 1, NEW_ITEM)
    else:
        # If Unit 3 not found (weird), just append
        data.append(NEW_ITEM)

    with open(JSON_PATH, 'w') as f:
        json.dump(data, f, indent=4)

    print("Successfully added Semi-Supervised Learning tutorial.")

if __name__ == "__main__":
    add_tutorial()
