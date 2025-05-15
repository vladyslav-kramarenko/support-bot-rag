import os
import requests

def download_google_doc(file_id: str, filename: str, output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)
    url = f"https://docs.google.com/document/d/{file_id}/export?format=txt"
    output_path = os.path.join(output_dir, filename)

    if os.path.exists(output_path):
        os.remove(output_path)

    response = requests.get(url)
    if response.status_code == 200:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(response.text)
        print(f"✅ Google Doc saved to: {output_path}")
    else:
        raise RuntimeError(f"❌ Failed to download Google Doc: {response.status_code}")

    return output_path