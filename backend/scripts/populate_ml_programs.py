import json
import re
import os

RAW_FILE = r'd:/satvik/ANTGVTPROJECT/PYProj/TutorialPlatform/backend/data/ml_practicals_raw.txt'
OUTPUT_FILE = r'd:/satvik/ANTGVTPROJECT/PYProj/TutorialPlatform/backend/data/programs.json'

def parse_practicals(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Regex to split by "📌 PRACTICAL – <number>"
    # Groups: 1=ID, 2=Title, 3=Code
    # Note: Code spans multiple lines until next practical or end of string
    pattern = re.compile(r'📌 PRACTICAL – (\d+)\n(.*?)\n([\s\S]*?)(?=📌 PRACTICAL|$)')
    matches = pattern.findall(content)
    
    parsed_data = []
    
    for match in matches:
        p_id = int(match[0])
        title = match[1].strip()
        code = match[2].strip()
        
        # Categorize into Units
        unit = "Unit 1: Introduction to ML" # Default
        if 1 <= p_id <= 15 or 19 <= p_id <= 22:
            unit = "Unit 1: Introduction to ML"
        elif 16 <= p_id <= 18 or 23 <= p_id <= 30:
            unit = "Unit 2: Supervised Learning"
        elif 31 <= p_id <= 37:
            unit = "Unit 3: Unsupervised Learning"
        elif 38 <= p_id <= 44:
            unit = "Unit 4: NLP"
        elif 45 <= p_id <= 51:
            unit = "Unit 5: Computer Vision"
            
        parsed_data.append({
            "course": "Machine Learning",
            "unit": unit,
            "question": f"{title}",
            "code": code
        })
        
    return parsed_data

def update_json(new_items):
    if not os.path.exists(OUTPUT_FILE):
        print("Error: programs.json not found.")
        return

    with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            data = []
            
    # Calculate new IDs
    max_id = 0
    if data:
        max_id = max(item.get('id', 0) for item in data)
        
    print(f"Current Max ID: {max_id}")
    
    count = 0
    for item in new_items:
        max_id += 1
        item['id'] = max_id
        data.append(item)
        count += 1
        
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
        
    print(f"Successfully added {count} ML programs. Final ID: {max_id}")

if __name__ == "__main__":
    if os.path.exists(RAW_FILE):
        items = parse_practicals(RAW_FILE)
        update_json(items)
    else:
        print(f"File not found: {RAW_FILE}")
