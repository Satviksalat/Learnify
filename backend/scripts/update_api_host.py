import os

SEARCH_DIR = r'd:/satvik/ANTGVTPROJECT/PYProj/TutorialPlatform/frontend/src'
OLD_HOST = "http://localhost:5004"
NEW_HOST = "http://192.168.5.138:5004"

def update_files():
    count = 0
    for root, dirs, files in os.walk(SEARCH_DIR):
        for file in files:
            if file.endswith(".jsx") or file.endswith(".js"):
                path = os.path.join(root, file)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    if OLD_HOST in content:
                        new_content = content.replace(OLD_HOST, NEW_HOST)
                        with open(path, 'w', encoding='utf-8') as f:
                            f.write(new_content)
                        print(f"Updated: {file}")
                        count += 1
                except Exception as e:
                    print(f"Error reading {file}: {e}")
    
    print(f"Total files updated: {count}")

if __name__ == "__main__":
    update_files()
