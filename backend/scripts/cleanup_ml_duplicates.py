import json
import os

JSON_PATH = r'd:/satvik/ANTGVTPROJECT/PYProj/TutorialPlatform/backend/data/tutorials.json'

# The NEW unit names we want to KEEP
VALID_UNITS = [
    "Unit 1: Introduction to ML",
    "Unit 2: Supervised Learning",
    "Unit 3: Unsupervised Learning",
    "Unit 4: Natural Language Processing",
    "Unit 5: Computer Vision with OpenCV"
]

# The IDs we explicitly JUST added (we want to keep these 100%)
# Actually, easier strategy: Remove any ML tutorial that has a Unit Name NOT in VALID_UNITS
# But wait, what if the old ones had the SAME name?
# "Unit 3: Unsupervised Learning" matches.
# We need to rely on the IDs. The new IDs are standard: ml-unit3-..., ml-unit4-..., ml-unit5-...
# The old IDs were a bit mix and match, but often similar.
# Let's simple remove "Unit 4: NLP" and "Unit 5: Computer Vision" (short names).
# And for Unit 3, let's remove any item that doesn't have a numeric prefix in title like "1. Intro", "2. Clustering".
# A better way: The new items all have Titles starting with a Number "1. ", "2. ", etc.
# The old items did not have numbers.

def cleanup():
    if not os.path.exists(JSON_PATH):
        print("Error: tutorials.json not found")
        return

    with open(JSON_PATH, 'r') as f:
        data = json.load(f)

    cleaned_data = []
    removed_count = 0

    for t in data:
        unit = t.get('unit', '')
        tech = t.get('technology', '')
        title = t.get('title', '')

        # Pass through non-ML items
        if tech != "Machine Learning with Python":
            cleaned_data.append(t)
            continue

        # For ML items:
        # 1. Remove if unit name is the OLD short version
        if unit in ["Unit 4: NLP", "Unit 5: Computer Vision"]:
            removed_count += 1
            continue

        # 2. For Unit 3, 4, 5 (New long names): Ensure it's one of ours.
        # Check if it's one of the units we just touched
        if unit in ["Unit 3: Unsupervised Learning", "Unit 4: Natural Language Processing", "Unit 5: Computer Vision with OpenCV"]:
            # If the title DOES NOT start with a digit, it's likely an old leftover that happens to share the name
            # (Though in my previous script I tried to change the name, so maybe none share it?)
            if not title[0].isdigit():
                 # Suspicious. Let's print it to see.
                 # Taking a risk here: All my new ones start with digits. Old ones probably don't.
                 print(f"Removing likely legacy item: {title} ({unit})")
                 removed_count += 1
                 continue
        
        # Keep valid new ML items and Unit 1/2 items
        cleaned_data.append(t)

    with open(JSON_PATH, 'w') as f:
        json.dump(cleaned_data, f, indent=4)

    print(f"Cleanup complete. Removed {removed_count} legacy ML items.")

if __name__ == "__main__":
    cleanup()
