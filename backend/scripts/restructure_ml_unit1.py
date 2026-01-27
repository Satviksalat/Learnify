import json
import os

JSON_PATH = r'd:/satvik/ANTGVTPROJECT/PYProj/TutorialPlatform/backend/data/tutorials.json'

NEW_STRUCTURE = [
    {
        "id": "ml-unit1-intro",
        "title": "1. What is Machine Learning?",
        "technology": "Machine Learning with Python",
        "unit": "Unit 1: Introduction to ML",
        "definition": "Formal Definitions & Analogies",
        "description": "Tom Mitchell definition, Why ML?",
        "syntax": "Theory",
        "code_example": "# Traditional: Data + Rules = Answers\n# ML: Data + Answers = Rules",
        "explanation": "Placeholder",
        "try_it_yourself": True,
        "key_points": ["Tom Mitchell", "Paradigm Shift", "No Explicit Programming"]
    },
    {
        "id": "ml-unit1-ai-vs-ml",
        "title": "2. AI vs ML vs DL",
        "technology": "Machine Learning with Python",
        "unit": "Unit 1: Introduction to ML",
        "definition": "The Hierarchy",
        "description": "Understanding the Venn Diagram.",
        "syntax": "Theory",
        "code_example": "# AI > ML > DL",
        "explanation": "Placeholder",
        "try_it_yourself": True,
        "key_points": ["Artificial Intelligence", "Deep Learning", "Data Science"]
    },
    {
        "id": "ml-unit1-howlearn",
        "title": "3. Learning from Data",
        "technology": "Machine Learning with Python",
        "unit": "Unit 1: Introduction to ML",
        "definition": "Pattern Recognition",
        "description": "Training vs Testing concepts.",
        "syntax": "Theory",
        "code_example": "model.fit(train_data)\nmodel.predict(test_data)",
        "explanation": "Placeholder",
        "try_it_yourself": True,
        "key_points": ["Generalization", "Overfitting", "Model Model"]
    },
    {
        "id": "ml-unit1-steps",
        "title": "4. The 7 Steps of ML",
        "technology": "Machine Learning with Python",
        "unit": "Unit 1: Introduction to ML",
        "definition": "The Workflow",
        "description": "From Data Collection to Prediction.",
        "syntax": "Theory",
        "code_example": "# 1. Gather\n# 2. Prepare\n# 3. Choose Model...",
        "explanation": "Placeholder",
        "try_it_yourself": True,
        "key_points": ["Preprocessing", "Training", "Evaluation"]
    },
    {
        "id": "ml-unit1-types",
        "title": "5. Types of Learning (Overview)",
        "technology": "Machine Learning with Python",
        "unit": "Unit 1: Introduction to ML",
        "definition": "Broad Categories",
        "description": "Supervised, Unsupervised, RL.",
        "syntax": "Theory",
        "code_example": "print('The Big Three')",
        "explanation": "Placeholder",
        "try_it_yourself": True,
        "key_points": ["Supervised", "Unsupervised", "Reinforcement"]
    },
    {
        "id": "ml-unit1-supervised-theory",
        "title": "6. Supervised Learning",
        "technology": "Machine Learning with Python",
        "unit": "Unit 1: Introduction to ML",
        "definition": "Labeled Data",
        "description": "Classification vs Regression theory.",
        "syntax": "Theory",
        "code_example": "# Input (X) + Output (Y) -> Map",
        "explanation": "Placeholder",
        "try_it_yourself": True,
        "key_points": ["Labels", "Regression", "Classification"]
    },
    {
        "id": "ml-unit1-unsupervised-theory",
        "title": "7. Unsupervised Learning",
        "technology": "Machine Learning with Python",
        "unit": "Unit 1: Introduction to ML",
        "definition": "Unlabeled Data",
        "description": "Clustering and Dimensionality Reduction.",
        "syntax": "Theory",
        "code_example": "# Input (X) -> Structure",
        "explanation": "Placeholder",
        "try_it_yourself": True,
        "key_points": ["Clustering", "Hidden Patterns", "No Teacher"]
    },
    {
        "id": "ml-unit1-rl-theory",
        "title": "8. Reinforcement Learning",
        "technology": "Machine Learning with Python",
        "unit": "Unit 1: Introduction to ML",
        "definition": "Trial and Error",
        "description": "Agent, Environment, Rewards.",
        "syntax": "Theory",
        "code_example": "# Action -> Reward -> Update Policy",
        "explanation": "Placeholder",
        "try_it_yourself": True,
        "key_points": ["Agent", "Reward Signal", "Policy"]
    },
    {
        "id": "ml-unit1-applications",
        "title": "9. Real-World Applications",
        "technology": "Machine Learning with Python",
        "unit": "Unit 1: Introduction to ML",
        "definition": "Industry Use Cases",
        "description": "Healthcare, Finance, Transport.",
        "syntax": "Theory",
        "code_example": "# Diagnosis, Fraud Detection, Self-Driving",
        "explanation": "Placeholder",
        "try_it_yourself": True,
        "key_points": ["Healthcare", "Finance", "Social Media"]
    },
    {
        "id": "ml-unit1-summary",
        "title": "10. Unit Summary",
        "technology": "Machine Learning with Python",
        "unit": "Unit 1: Introduction to ML",
        "definition": "Recap & Interview Prep",
        "description": "Key takeaways and common questions.",
        "syntax": "Theory",
        "code_example": "# Ready for Unit 2!",
        "explanation": "Placeholder",
        "try_it_yourself": True,
        "key_points": ["Definitions", "Differences", "Examples"]
    }
]

def restructure():
    if not os.path.exists(JSON_PATH):
        print("Error: tutorials.json not found")
        return

    with open(JSON_PATH, 'r') as f:
        data = json.load(f)

    # Remove ALL existing ML Unit 1 items to prevent duplicates
    # We identify them by unit name OR by id prefix 'ml-unit1-'
    cleaned_data = [t for t in data if t.get('unit') != "Unit 1: Introduction to ML"]

    # Insert the new block where the old block roughly was (or just append, but let's try to put it before Unit 2)
    # Find index of first Unit 2 item
    insert_index = len(cleaned_data)
    for i, t in enumerate(cleaned_data):
        if t.get('unit') == "Unit 2: Supervised Learning":
            insert_index = i
            break
    
    # Insert new structure
    for item in reversed(NEW_STRUCTURE):
        cleaned_data.insert(insert_index, item)

    with open(JSON_PATH, 'w') as f:
        json.dump(cleaned_data, f, indent=4)

    print("Successfully restructured ML Unit 1.")

if __name__ == "__main__":
    restructure()
