import os
import json
import subprocess

def setup_kaggle_and_download():
    print("=== Tải dữ liệu Face Shape Dataset từ Kaggle ===")
    
    # Kiểm tra kaggle.json
    kaggle_json_path = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_json_path):
        print("Không tìm thấy ~/.kaggle/kaggle.json")
        print("Vui lòng thực hiện các bước sau:")
        print("1. Đăng nhập vào Kaggle.com -> Account -> Create New API Token")
        print("2. Chép file kaggle.json tải về vào thư mục ~/.kaggle/")
        print("3. Chạy lệnh: chmod 600 ~/.kaggle/kaggle.json")
        print("Sau đó chạy lại script này.")
        return

    # Tải dataset
    dataset_name = "niten19/face-shape-dataset"
    download_dir = "data/face-shape-dataset"
    
    if os.path.exists(download_dir):
        print(f"Thư mục {download_dir} đã tồn tại. Bỏ qua việc tải dữ liệu.")
        return
        
    os.makedirs("data", exist_ok=True)
    print(f"Đang tải dataset {dataset_name}...")
    
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset_name, "-p", "data", "--unzip"],
            check=True
        )
        print("Tải và giải nén dữ liệu thành công vào thư mục: data/face-shape-dataset")
        # Rename the extracted folder to a standard name if needed
        # Kaggle might extract to face-shape-dataset or FaceShape Dataset
        # The script will just use whatever kaggle extracts.
    except Exception as e:
        print(f"Lỗi tải dữ liệu: {e}")

if __name__ == "__main__":
    setup_kaggle_and_download()
