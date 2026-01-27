import json
import os

JSON_PATH = r'd:/satvik/ANTGVTPROJECT/PYProj/TutorialPlatform/backend/data/exam_questions.json'

NEW_QUESTIONS = {
    "Part A (1-Mark)": [
        {"question": "Define Semi-Supervised Learning.", "answer": "A machine learning approach that combines a small amount of labeled data with a large amount of unlabeled data during training."},
        {"question": "Give one example of Semi-Supervised Learning.", "answer": "Self-Training, where a model trains on labeled data, labels the unlabeled data, and retrains itself."}
    ],
    "Part B (5-Marks)": [
        {"question": "Why is Semi-Supervised Learning useful? Explain with an analogy.", "answer": "1. Cost Efficiency: Labeling data is expensive (time/money). Unlabeled data is cheap.\n2. Accuracy: It performs better than Unsupervised learning and cheaper than Supervised.\n3. Analogy: A teacher solves 5 difficult logic puzzles (Labeled) for the class. The students then use that logic to solve 95 similar unsolved puzzles (Unlabeled) on their own."}
    ],
    "Part C (10-Marks)": [
        {"question": "Compare Supervised, Unsupervised, and Semi-Supervised Learning.", "answer": "1. Supervised:\n- Input: Fully Labeled Data.\n- Goal: Predict outcomes/classify.\n- Cost: High (labeling).\n\n2. Unsupervised:\n- Input: No Labels.\n- Goal: Find hidden structure/patterns.\n- Cost: Low (data is raw).\n\n3. Semi-Supervised:\n- Input: Mix of Labeled (Small) + Unlabeled (Large).\n- Goal: Improve accuracy using cheap data.\n- Cost: Medium (Best of both worlds).\n- Use Case: Medical Imaging where doctors (labelers) are expensive."}
    ]
}

def add_questions():
    if not os.path.exists(JSON_PATH):
        print("Error: exam_questions.json not found")
        return

    with open(JSON_PATH, 'r') as f:
        data = json.load(f)

    unit_found = False
    for unit in data:
        if unit.get("unit") == "Unit 3: Unsupervised Learning":
            unit_found = True
            sections = unit.get("sections", {})
            
            # Append questions
            for section_name, questions in NEW_QUESTIONS.items():
                if section_name in sections:
                    sections[section_name].extend(questions)
                else:
                    sections[section_name] = questions
            
            print("Added Semi-Supervised questions to Unit 3.")
            break
    
    if not unit_found:
        print("Error: Unit 3 not found in exam_questions.json")

    with open(JSON_PATH, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    add_questions()
