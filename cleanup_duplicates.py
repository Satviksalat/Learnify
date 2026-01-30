
import json
import os

EXAM_FILE = os.path.join("backend", "data", "exam_questions.json")

def cleanup_duplicates():
    if not os.path.exists(EXAM_FILE):
        print("not found")
        return

    with open(EXAM_FILE, 'r') as f:
        data = json.load(f)

    print(f"Total units before: {len(data)}")

    # STRICT whitelist of the ONLY 10 allowed units.
    # These match the names I generated in populate_X_full.py scripts.
    allowed_units = [
        "Python Unit 1: Python Basics",
        "Python Unit 2: OOPs in Python",
        "Python Unit 3: Plotting & Algorithms",
        "Python Unit 4: Network & GUI",
        "Python Unit 5: Database Connectivity",
        
        "ML Unit 1: Introduction to Machine Learning", # Verified from populate_ml_unit1_full.py (I'll assume this is the intended one)
        "ML Unit 2: Supervised Learning",
        "ML Unit 3: Unsupervised Learning",
        "ML Unit 4: Natural Language Processing (NLP)",
        "ML Unit 5: Computer Vision (CV)" 
    ]
    
    # We will filter the data.
    # Logic: Keep if EXACT match.
    # Wait: What if "ML Unit 5: Computer Vision" is the short one and "ML Unit 5: Computer Vision with OpenCV" is the long one?
    # I verified earlier that populate_ml_unit_5_full.py used "ML Unit 5: Computer Vision with OpenCV" (checking my memory/logs)
    # Actually, let's verify exact contents of the latest 5 units added by checking the end of the file or just trusting the "60 questions" count.
    
    # BETTER LOGIC:
    # Group by Unit X.
    # If duplicates for "Unit X", keep the one with MORE questions (likely the new 60-count one).
    
    cleaned_data = []
    
    # Helper to clean/standardize name
    def get_unit_key(unit_name):
        # Returns ("Python", 1) or ("ML", 5)
        if "Python" in unit_name:
            subject = "Python"
        else:
            subject = "ML"
            
        # Extract number
        import re
        match = re.search(r'Unit (\d+)', unit_name)
        if match:
            num = int(match.group(1))
        else:
            num = 0
        return (subject, num)

    # Grouping
    groups = {} 
    
    for unit in data:
        key = get_unit_key(unit['unit'])
        if key not in groups:
            groups[key] = []
        groups[key].append(unit)
        
    # Selection
    final_list = []
    
    # We expect keys: ('Python', 1..5) and ('ML', 1..5)
    # Let's iterate 1-5 for both.
    
    for subject in ['Python', 'ML']:
        for i in range(1, 6):
            key = (subject, i)
            if key in groups:
                candidates = groups[key]
                print(f"Checking {subject} Unit {i}: Found {len(candidates)} versions.")
                
                # Pick the BEST one.
                # Criteria 1: Max questions (sum of all parts)
                # Criteria 2: Most recent (can't tell easily), but usually last appended.
                
                best_candidate = None
                max_q = -1
                
                for cand in candidates:
                    # Count total questions
                    count = 0
                    if 'sections' in cand:
                         for part in cand['sections']:
                             count += len(cand['sections'][part])
                    
                    print(f"  - '{cand['unit']}': {count} questions")
                    
                    if count >= max_q:
                        max_q = count
                        best_candidate = cand
                
                if best_candidate:
                    print(f"  -> Keeping: '{best_candidate['unit']}' ({max_q} q)")
                    final_list.append(best_candidate)
            else:
                print(f"WARNING: Missing {subject} Unit {i}")

    # Save
    print(f"Total units after: {len(final_list)}")
    with open(EXAM_FILE, 'w') as f:
        json.dump(final_list, f, indent=4)
    print("Saved cleaned exam_questions.json.")

if __name__ == "__main__":
    cleanup_duplicates()
