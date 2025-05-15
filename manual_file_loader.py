import gdown
import os

def download_file_from_drive(file_id: str, filename: str, output_dir='data'):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    # Delete old file, if exists
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"🗑 Удалён старый файл: {output_path}")

    # Download new file
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, output_path, quiet=False)
    print(f"✅ Загружен новый файл: {output_path}")