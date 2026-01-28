import os

# Configuration
OLD_URL = "http://192.168.5.138:5004"
NEW_URL = "https://learnify-api-ohc0.onrender.com"
FRONTEND_DIR = r"d:/satvik/ANTGVTPROJECT/PYProj/TutorialPlatform/frontend/src"

def update_files():
    count = 0
    for root, dirs, files in os.walk(FRONTEND_DIR):
        for file in files:
            if file.endswith(".jsx") or file.endswith(".js"):
                file_path = os.path.join(root, file)
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    if OLD_URL in content:
                        new_content = content.replace(OLD_URL, NEW_URL)
                        
                        with open(file_path, 'w', encoding='utf-8') as f:
                            f.write(new_content)
                        
                        print(f"Updated: {file}")
                        count += 1
                except Exception as e:
                    print(f"Error processing {file}: {e}")

    print(f"\nTotal files updated: {count}")

if __name__ == "__main__":
    update_files()
