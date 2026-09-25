import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from models.cart import CART
from data.loaders import load_openml_dataset, apply_synthetic_corruption
from experiments.main_benchmark_table4 import PairedTabularDataset, set_seed
from sklearn.metrics import accuracy_score

def get_ablation_logits_and_trust(model, x_n_cr, x_c_cr, x_n_cl, x_c_cl, mask_true, mode):
    device = x_n_cr.device
    batch_size = x_n_cr.shape[0]
    
    # 1. Nhúng dữ liệu
    z = model.tokenizer(x_n_cr, x_c_cr)
    z_clean = model.tokenizer(x_n_cl, x_c_cl) if mode == 'mask_oracle' else None
        
    if mode == 'raw':
        z_star = z
        r_out = None
    else:
        # Nhánh Repair (True LOO)
        f_ids = torch.arange(model.total_features, device=device).unsqueeze(0).expand(batch_size, -1)
        f_embs = model.tokenizer.feature_id(f_ids)
        c_repair = model.repair_context(z, query_tokens=f_embs)
        z_repaired = model.repair_net(torch.cat([c_repair, f_embs], dim=-1))
        
        # Nhánh Trust (chỉ lấy giá trị nếu là các mode cần thiết)
        c_trust = z
        for block in model.trust_context: c_trust = block(c_trust)
        trust_logits = model.trust_net(torch.cat([z, c_trust], dim=-1)).squeeze(-1)
        r_actual = torch.sigmoid(trust_logits)
        
        if mode == 'repair_only':
            z_star = z_repaired
            r_out = None
            
        elif mode == 'mask_oracle':
            # MASK ORACLE: Dùng z_clean cho các ô bị hỏng
            m_expanded = mask_true.unsqueeze(-1)
            z_star = m_expanded * z + (1.0 - m_expanded) * z_clean
            r_out = None
            
        elif mode == 'fixed_fusion':
            # Giả định trung bình dữ liệu có 20% nhiễu.
            r_fixed = torch.full((batch_size, model.total_features), 0.8, device=device)
            r_expanded = r_fixed.unsqueeze(-1)
            z_star = r_expanded * z + (1.0 - r_expanded) * z_repaired
            r_out = None
            
        elif mode == 'random_uniform':
            # Random phá vỡ cả feature assignment lẫn phân phối
            r_random = torch.rand(batch_size, model.total_features, device=device)
            r_expanded = r_random.unsqueeze(-1)
            z_star = r_expanded * z + (1.0 - r_expanded) * z_repaired
            r_out = None
            
        elif mode == 'permuted_trust':
            idx = torch.randperm(model.total_features, device=device)
            r_permuted = r_actual[:, idx]
            r_expanded = r_permuted.unsqueeze(-1)
            z_star = r_expanded * z + (1.0 - r_expanded) * z_repaired
            r_out = None
            
        elif mode == 'cart_dynamic':
            # Dùng chính xác r_actual cho từng ô
            r_expanded = r_actual.unsqueeze(-1)
            z_star = r_expanded * z + (1.0 - r_expanded) * z_repaired
            r_out = r_actual # Chỉ xuất r_actual ở mode này để tính Mean Trust

    # Đưa qua bộ phân loại
    cls_tokens = model.cls_token.expand(batch_size, -1, -1)
    c_star = torch.cat([cls_tokens, z_star], dim=1)
    for block in model.pred_encoder: 
        c_star = block(c_star)
        
    return model.classifier(c_star[:, 0, :]), r_out

def main():
    EXP_VERSION = "main_results_table4"
    datasets = ['Adult', 'HELOC', 'Higgs', 'Bank', 'Credit', 'Churn', 'Diabetes', 'Magic']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    severities = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    results = []
    print("=== ABLATION 7-WAY ===")
    
    for d_name in datasets:
        try:
            _, (X_te_n_raw, X_te_c_raw, y_te_raw), card = load_openml_dataset(d_name)
        except: continue
        num_num = X_te_n_raw.shape[1]
        
        for seed in [42, 123, 2024]:
            set_seed(seed)
            
            # Khớp RNG 100% với Benchmark
            target_ev_n, target_ev_c, target_mask = None, None, None
            for rate in severities:
                if rate == 0.0: continue
                ev_n, ev_c, m_ev = apply_synthetic_corruption(X_te_n_raw, X_te_c_raw, card, rate)
                if rate == 0.2:
                    target_ev_n, target_ev_c, target_mask = ev_n, ev_c, m_ev
            
            dl_eval = DataLoader(PairedTabularDataset(X_te_n_raw, X_te_c_raw, target_ev_n, target_ev_c, y_te_raw, target_mask), batch_size=256, shuffle=False)
            
            # Chỉ dùng duy nhất mô hình CART Dynamic đã train chuẩn
            model = CART(num_num, card).to(device)
            ckpt = f"checkpoints/{EXP_VERSION}/seed{seed}/{d_name}_CART-v8-Dynamic.pt"
            if not os.path.exists(ckpt): continue
            model.load_state_dict(torch.load(ckpt, map_location=device))
            model.eval()
            
            p_raw, p_rep, p_fixed, p_uni, p_perm, p_dyn, p_oracle, all_y = [], [], [], [], [], [], [], []
            all_r_clean, all_r_corrupt = [], []
            
            with torch.no_grad():
                for batch in dl_eval:
                    xn_cl, xc_cl, xn_cr, xc_cr, y_t, m_true = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[3].to(device), batch[4].to(device), batch[5].to(device)
                    
                    logits_raw, _ = get_ablation_logits_and_trust(model, xn_cr, xc_cr, xn_cl, xc_cl, m_true, 'raw')
                    logits_rep, _ = get_ablation_logits_and_trust(model, xn_cr, xc_cr, xn_cl, xc_cl, m_true, 'repair_only')
                    logits_fixed, _ = get_ablation_logits_and_trust(model, xn_cr, xc_cr, xn_cl, xc_cl, m_true, 'fixed_fusion')
                    logits_uni, _ = get_ablation_logits_and_trust(model, xn_cr, xc_cr, xn_cl, xc_cl, m_true, 'random_uniform')
                    logits_perm, _ = get_ablation_logits_and_trust(model, xn_cr, xc_cr, xn_cl, xc_cl, m_true, 'permuted_trust')
                    logits_oracle, _ = get_ablation_logits_and_trust(model, xn_cr, xc_cr, xn_cl, xc_cl, m_true, 'mask_oracle')
                    
                    # Lấy Trust để tính thống kê
                    logits_dyn, r_actual = get_ablation_logits_and_trust(model, xn_cr, xc_cr, xn_cl, xc_cl, m_true, 'cart_dynamic')
                    
                    # Thống kê Trust theo mask (1 = clean, 0 = corrupt)
                    r_clean = r_actual[m_true == 1.0].cpu().numpy()
                    r_corrupt = r_actual[m_true == 0.0].cpu().numpy()
                    all_r_clean.extend(r_clean)
                    all_r_corrupt.extend(r_corrupt)
                    
                    p_raw.extend(torch.argmax(logits_raw, dim=1).cpu().numpy())
                    p_rep.extend(torch.argmax(logits_rep, dim=1).cpu().numpy())
                    p_fixed.extend(torch.argmax(logits_fixed, dim=1).cpu().numpy())
                    p_uni.extend(torch.argmax(logits_uni, dim=1).cpu().numpy())
                    p_perm.extend(torch.argmax(logits_perm, dim=1).cpu().numpy())
                    p_dyn.extend(torch.argmax(logits_dyn, dim=1).cpu().numpy())
                    p_oracle.extend(torch.argmax(logits_oracle, dim=1).cpu().numpy())
                    all_y.extend(y_t.cpu().numpy())
            
            res = {
                'Dataset': d_name, 'Seed': seed,
                'Raw': accuracy_score(all_y, p_raw) * 100,
                'Repair_Only': accuracy_score(all_y, p_rep) * 100,
                'Fixed_80_20': accuracy_score(all_y, p_fixed) * 100,
                'Random_Uniform': accuracy_score(all_y, p_uni) * 100,
                'Permuted_Trust': accuracy_score(all_y, p_perm) * 100,
                'Dynamic_CART': accuracy_score(all_y, p_dyn) * 100,
                'Mask_Oracle': accuracy_score(all_y, p_oracle) * 100,
                'MeanTrust_Clean': np.mean(all_r_clean),
                'MeanTrust_Corrupt': np.mean(all_r_corrupt)
            }
            results.append(res)
            print(f"[{d_name}-{seed}] Raw:{res['Raw']:.1f} | Fixed:{res['Fixed_80_20']:.1f} | Perm:{res['Permuted_Trust']:.1f} | Dyn:{res['Dynamic_CART']:.1f} || Trust_Cln: {res['MeanTrust_Clean']:.2f} - Trst_Cor: {res['MeanTrust_Corrupt']:.2f}")

    df = pd.DataFrame(results)
    metric_cols = ['Raw', 'Repair_Only', 'Fixed_80_20', 'Random_Uniform', 'Permuted_Trust', 'Dynamic_CART', 'Mask_Oracle', 'MeanTrust_Clean', 'MeanTrust_Corrupt']
    df_agg = df.groupby(['Dataset'])[metric_cols].mean().reset_index()
    
    df.to_csv("results/ablation_7way_final_raw.csv", index=False)
    df_agg.to_csv("results/ablation_7way_final_agg.csv", index=False)
    print("\nĐã được lưu tại results/ablation_7way_final_agg.csv")

if __name__ == "__main__":
    main()