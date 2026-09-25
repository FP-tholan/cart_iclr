import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from models.cart import CART
from data.loaders import load_openml_dataset, apply_synthetic_corruption
from experiments.main_benchmark_table4 import PairedTabularDataset, set_seed
from sklearn.metrics import accuracy_score

def get_oracle_budget_logits(model, x_n_cr, x_c_cr, mask_true, mode):
    device = x_n_cr.device
    batch_size = x_n_cr.shape[0]
    z = model.tokenizer(x_n_cr, x_c_cr)
    
    f_ids = torch.arange(model.total_features, device=device).unsqueeze(0).expand(batch_size, -1)
    f_embs = model.tokenizer.feature_id(f_ids)
    c_repair = model.repair_context(z, query_tokens=f_embs)
    z_repaired = model.repair_net(torch.cat([c_repair, f_embs], dim=-1))
    
    c_trust = z
    for block in model.trust_context: c_trust = block(c_trust)
    r_actual = torch.sigmoid(model.trust_net(torch.cat([z, c_trust], dim=-1)).squeeze(-1))
    
    # Tìm số lượng ô hỏng cho từng sample
    k_per_sample = (1.0 - mask_true).sum(dim=1).long() # mask=0 là hỏng
    
    intervention_mask = torch.ones_like(r_actual)
    
    for i in range(batch_size):
        k = k_per_sample[i].item()
        if k > 0:
            if mode == 'low_trust_k':
                _, indices = torch.topk(r_actual[i], k=k, largest=False)
                intervention_mask[i].scatter_(0, indices, 0.0)
            elif mode == 'high_trust_k':
                _, indices = torch.topk(r_actual[i], k=k, largest=True)
                intervention_mask[i].scatter_(0, indices, 0.0)
            elif mode == 'random_k':
                rand_scores = torch.rand_like(r_actual[i])
                _, indices = torch.topk(rand_scores, k=k)
                intervention_mask[i].scatter_(0, indices, 0.0)
            elif mode == 'true_corrupted_k': # Baseline Oracle cho intervention
                intervention_mask[i] = mask_true[i]

    m_expanded = intervention_mask.unsqueeze(-1)
    z_star = m_expanded * z + (1.0 - m_expanded) * z_repaired

    cls_tokens = model.cls_token.expand(batch_size, -1, -1)
    c_star = torch.cat([cls_tokens, z_star], dim=1)
    for block in model.pred_encoder: c_star = block(c_star)
    return model.classifier(c_star[:, 0, :])

def main():
    EXP_VERSION = "main_results_table4"
    datasets = ['Adult', 'HELOC', 'Higgs', 'Bank', 'Credit', 'Churn', 'Diabetes', 'Magic']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    results = []
    print("=== TARGETED REPAIR (ORACLE BUDGET k) ===")
    
    for d_name in datasets:
        try:
            _, (X_te_n_raw, X_te_c_raw, y_te_raw), card = load_openml_dataset(d_name)
        except: continue
        num_num = X_te_n_raw.shape[1]
        
        for seed in [42, 123, 2024]:
            set_seed(seed)
            ev_n, ev_c, m_ev = apply_synthetic_corruption(X_te_n_raw, X_te_c_raw, card, 0.2)
            dl_eval = DataLoader(PairedTabularDataset(X_te_n_raw, X_te_c_raw, ev_n, ev_c, y_te_raw, m_ev), batch_size=256)
            
            model = CART(num_num, card).to(device)
            ckpt = f"checkpoints/{EXP_VERSION}/seed{seed}/{d_name}_CART-v8-Dynamic.pt"
            if not os.path.exists(ckpt): continue
            model.load_state_dict(torch.load(ckpt, map_location=device))
            model.eval()
            
            p_raw, p_low, p_rand, p_high, p_true, all_y = [], [], [], [], [], []
            with torch.no_grad():
                for batch in dl_eval:
                    xn_cr, xc_cr, yt, m_true = batch[2].to(device), batch[3].to(device), batch[4].to(device), batch[5].to(device)
                    
                    logits_raw = get_oracle_budget_logits(model, xn_cr, xc_cr, m_true, 'raw') # K = 0 (tất cả là mask=1)
                    logits_low = get_oracle_budget_logits(model, xn_cr, xc_cr, m_true, 'low_trust_k')
                    logits_rand = get_oracle_budget_logits(model, xn_cr, xc_cr, m_true, 'random_k')
                    logits_high = get_oracle_budget_logits(model, xn_cr, xc_cr, m_true, 'high_trust_k')
                    logits_true = get_oracle_budget_logits(model, xn_cr, xc_cr, m_true, 'true_corrupted_k')
                    
                    p_raw.extend(torch.argmax(logits_raw, dim=1).cpu().numpy())
                    p_low.extend(torch.argmax(logits_low, dim=1).cpu().numpy())
                    p_rand.extend(torch.argmax(logits_rand, dim=1).cpu().numpy())
                    p_high.extend(torch.argmax(logits_high, dim=1).cpu().numpy())
                    p_true.extend(torch.argmax(logits_true, dim=1).cpu().numpy())
                    all_y.extend(yt.cpu().numpy())
            
            res = {
                'Dataset': d_name, 'Seed': seed,
                'Raw': accuracy_score(all_y, p_raw) * 100,
                'HighTrust_k': accuracy_score(all_y, p_high) * 100,
                'Random_k': accuracy_score(all_y, p_rand) * 100,
                'LowTrust_k': accuracy_score(all_y, p_low) * 100,
                'TrueCorrupted_k': accuracy_score(all_y, p_true) * 100
            }
            results.append(res)
            print(f"[{d_name}-{seed}] Raw:{res['Raw']:.1f} | High:{res['HighTrust_k']:.1f} | Rand:{res['Random_k']:.1f} | Low:{res['LowTrust_k']:.1f} | True(Oracle):{res['TrueCorrupted_k']:.1f}")

    df = pd.DataFrame(results)
    df_agg = df.groupby(['Dataset']).mean().reset_index()
    df_agg.to_csv("results/targeted_repair_oracle_agg.csv", index=False)
    print("\nNếu LowTrust ~ TrueCorrupted -> ngon.")

if __name__ == "__main__":
    main()