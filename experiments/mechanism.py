import os
import torch
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import IsolationForest
from torch.utils.data import DataLoader

from models.cart import CART
from baselines.catboost_base import CatBoostBaseline
from data.loaders import load_openml_dataset, apply_synthetic_corruption, TabularDataset
from evaluation.metrics import calc_cell_detection
from experiments.main_benchmark_table4 import set_seed

def extract_trust_scores(cart_model, loader, device):
    cart_model.eval()
    all_trust = []
    all_masks = []
    with torch.no_grad():
        for xn, xc, _, mask in loader:
            xn, xc = xn.to(device), xc.to(device)
            out = cart_model(xn, xc)
            trust = out[2] 
            all_trust.append(trust.cpu().numpy())
            all_masks.append(mask.numpy())
    return np.vstack(all_trust), np.vstack(all_masks)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EXP_VERSION = "main_results_table4"
    d_name = 'Adult'
    seed = 42
    set_seed(seed)
    
    (X_tr_n, X_tr_c, y_tr), (X_te_n, X_te_c, y_te), card = load_openml_dataset(d_name)
    num_num = X_tr_n.shape[1]
    
    X_te_n_c, X_te_c_c, mask_te = apply_synthetic_corruption(X_te_n, X_te_c, card, corruption_rate=0.2)
    
    print("\n[1/3] Đang chạy Isolation Forest (Anomaly Detection)...")
    X_te_full = np.concatenate([X_te_n_c, X_te_c_c], axis=1)
    iso_forest = IsolationForest(random_state=seed, n_jobs=-1)
    iso_forest.fit(X_te_full)
    anomaly_scores_sample = -iso_forest.decision_function(X_te_full) 
    
    print("[2/3] Đang tính SHAP Values từ CatBoost...")
    catb = CatBoostBaseline()
    catb.fit(X_tr_n, X_tr_c, y_tr)
    
    df_num_shap = pd.DataFrame(X_te_n_c[:500], dtype=float)
    df_cat_shap = pd.DataFrame(X_te_c_c[:500], dtype=int) 
    X_shap = pd.concat([df_num_shap, df_cat_shap], axis=1)
    X_shap.columns = [str(i) for i in range(X_shap.shape[1])]
    
    explainer = shap.TreeExplainer(catb.model)
    shap_values = explainer.shap_values(X_shap)
    importance_scores = np.abs(shap_values)

    print("[3/3] Đang lấy Trust Scores từ CART Checkpoint...")
    cart = CART(num_num, card).to(device)
    ckpt_path = f"checkpoints/{EXP_VERSION}/seed{seed}/{d_name}_CART-v8-Dynamic.pt"
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Không tìm thấy Checkpoint CART tại {ckpt_path}. Hãy đảm bảo bạn đã chạy benchmark trước!")
        
    cart.load_state_dict(torch.load(ckpt_path, map_location=device))
    cart.eval()
            
    test_loader_500 = DataLoader(TabularDataset(X_te_n_c[:500], X_te_c_c[:500], y_te[:500], mask_te[:500]), batch_size=500)
    trust_scores, true_masks = extract_trust_scores(cart, test_loader_500, device)
    
    auroc, auprc = calc_cell_detection(trust_scores, true_masks)
    print(f"\n[Metrics] Cell Detection AUROC: {auroc:.4f} | AUPRC: {auprc:.4f}")
    
    df_analysis = pd.DataFrame({
        'Is_Clean': true_masks.flatten(),
        'SHAP_Importance': importance_scores.flatten(),
        'CART_Trust': trust_scores.flatten(),
        'Sample_Anomaly': np.repeat(anomaly_scores_sample[:500], X_te_full.shape[1])
    })
    
    median_shap = df_analysis['SHAP_Importance'].median()
    q2_cells = df_analysis[(df_analysis['Is_Clean'] == 0) & 
                           (df_analysis['SHAP_Importance'] > median_shap) & 
                           (df_analysis['CART_Trust'] < 0.5)]
    
    print(f"\n[PHÁT HIỆN] Tìm thấy {len(q2_cells)} ô dữ liệu rơi vào Quadrant 2!")
    print("-> Đây là những ô rác, mang thông tin sai lệch, nhưng lại được đánh giá là RẤT QUAN TRỌNG bởi SHAP/CatBoost.")
    print("-> Nếu không có cơ chế gác cổng của CART, mô hình sẽ ăn trọn đống rác này và dự đoán sai lệch.")
    
    df_analysis.to_csv("results/mechanism_analysis.csv", index=False)
    print("\n[Hoàn tất] Đã lưu ra results/mechanism_analysis.csv")

if __name__ == "__main__":
    main()