import os
from sklearn.datasets import fetch_openml
import warnings
warnings.filterwarnings("ignore")

def pre_download_data(data_dir='./data'):
    os.makedirs(data_dir, exist_ok=True)
    
    datasets = {
        'Adult': 'adult',
        'HELOC': 'heloc',
        'Higgs': 23512, # ID bản subset của Higgs
        'Bank': 'bank-marketing',
        'Credit': 'default-of-credit-card-clients',
        'Churn': 'churn',
        'Diabetes': 'diabetes',
        'Magic': 'MagicTelescope'
    }
    
    print(f"=== BẮT ĐẦU KÉO {len(datasets)} DATASETS VỀ LOCAL CACHE ===")
    
    for name, fetch_id in datasets.items():
        print(f"Đang tải {name}...")
        try:
            if isinstance(fetch_id, int):
                fetch_openml(data_id=fetch_id, as_frame=True, parser='auto', data_home=data_dir)
            else:
                fetch_openml(name=fetch_id, version=1, as_frame=True, parser='auto', data_home=data_dir)
        except Exception:
            # Dự phòng version 2 nếu version 1 bị lỗi trên server
            fetch_openml(name=fetch_id, version=2, as_frame=True, parser='auto', data_home=data_dir)
            
        print(f" -> Đã lưu {name} thành công!")
        
    print("\n[Xong!] Toàn bộ data đã nằm gọn trong thư mục ./data")

if __name__ == "__main__":
    pre_download_data()