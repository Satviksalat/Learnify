
import json
import os

def verify_python_units():
    EXAM_FILE = os.path.join("backend", "data", "exam_questions.json")
    
    if not os.path.exists(EXAM_FILE):
        print("Error: exam_questions.json not found.")
        return

    with open(EXAM_FILE, 'r') as f:
        data = json.load(f)

    python_units = [
        "Python Unit 1: Python Basics",
        "Python Unit 2: OOPs in Python",
        "Python Unit 3: Plotting & Algorithms",
        "Python Unit 4: Network & GUI",
        "Python Unit 5: Database Connectivity"
    ]

    expected_counts = {
        "Part A (1-Mark)": 20,
        "Part B (2-Marks)": 15,
        "Part C (3-Marks)": 15,
        "Part D (5-Marks)": 10
    }

    all_good = True

    for target_unit in python_units:
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
        print("\nSUCCESS: All Python Units are present and have correct question counts.")
    else:
        print("\nFAILURE: Some units or counts are incorrect.")

if __name__ == "__main__":
    verify_python_units()
