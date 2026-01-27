import requests

BASE_URL = "http://localhost:5004/downloads"
FILES = [
    "python_cheat_sheet.md",
    "ml_quick_reference.md",
    "all_code_examples.json"
]

def verify_downloads():
    print("Verifying Downloadable Resources...")
    all_passed = True
    
    for filename in FILES:
        url = f"{BASE_URL}/{filename}"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                print(f"[PASS] {filename} is accessible. Size: {len(response.content)} bytes.")
            else:
                print(f"[FAIL] {filename} returned status {response.status_code}.")
                all_passed = False
        except Exception as e:
            print(f"[ERROR] Could not connect to {url}: {e}")
            all_passed = False
            
    if all_passed:
        print("\nSUCCESS: All download links are working correctly.")
    else:
        print("\nFAILURE: Some download links are broken.")

if __name__ == "__main__":
    verify_downloads()
