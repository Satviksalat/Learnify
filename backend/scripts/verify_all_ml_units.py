
import json
import os

EXAM_FILE = os.path.join("backend", "data", "exam_questions.json")

def verify_ml_units():
    try:
        if not os.path.exists(EXAM_FILE):
             print("Error: exam_questions.json not found.")
             return

        with open(EXAM_FILE, 'r') as f:
            data = json.load(f)

        expected_counts = {
            "Part A (1-Mark)": 20,
            "Part B (2-Marks)": 15,
            "Part C (3-Marks)": 15,
            "Part D (5-Marks)": 10
        }

        ml_units = [
            "ML Unit 1: Introduction to ML",
            "ML Unit 2: Supervised Learning",
            "ML Unit 3: Unsupervised Learning",
            "ML Unit 4: Natural Language Processing",
            "ML Unit 5: Computer Vision with OpenCV"
        ]

        all_good = True

        for target_unit in ml_units:
            found = False
            for unit in data:
                if unit['unit'] == target_unit:
                    found = True
                    print(f"\nVerifying {target_unit}...")
                    for section, count in expected_counts.items():
                        actual = len(unit['sections'].get(section, []))
                        if actual == count:
                            print(f"  - {section}: {actual} [OK]")
                        else:
                            print(f"  - {section}: {actual} [FAIL] (Expected {count})")
                            all_good = False
                    break
            
            if not found:
                print(f"Error: {target_unit} NOT FOUND in database.")
                all_good = False
        
        if all_good:
            print("\nSUCCESS: All ML Units are present and have correct question counts.")
        else:
            print("\nFAILURE: Some units or counts are incorrect.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    verify_ml_units()
