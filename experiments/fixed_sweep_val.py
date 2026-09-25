import os
import torch
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from models.cart import CART
from data.loaders import load_openml_dataset, apply_synthetic_corruption
from experiments.main_benchmark_table4 import PairedTabularDataset, set_seed

def get_sweep_logits(model, x_n_cr, x_c_cr, fixed_r=None):
    device = x_n_cr.device
    batch_size = x_n_cr.shape[0]
    z = model.tokenizer(x_n_cr, x_c_cr)
    
    f_ids = torch.arange(model.total_features, device=device).unsqueeze(0).expand(batch_size, -1)
    f_embs = model.tokenizer.feature_id(f_ids)
    c_repair = model.repair_context(z, query_tokens=f_embs)
    z_repaired = model.repair_net(torch.cat([c_repair, f_embs], dim=-1))
    
    if fixed_r is not None:
        r = torch.full((batch_size, model.total_features), fixed_r, device=device)
    else:
        c_trust = z
        for block in model.trust_context: c_trust = block(c_trust)
        r = torch.sigmoid(model.trust_net(torch.cat([z, c_trust], dim=-1)).squeeze(-1))
        
    r_expanded = r.unsqueeze(-1)
    z_star = r_expanded * z + (1.0 - r_expanded) * z_repaired
    
    cls_tokens = model.cls_token.expand(batch_size, -1, -1)
    c_star = torch.cat([cls_tokens, z_star], dim=1)
    for block in model.pred_encoder: c_star = block(c_star)
    return model.classifier(c_star[:, 0, :])
def main():
    EXP_VERSION = "main_results_table4"
    datasets = ['Adult', 'HELOC', 'Higgs', 'Bank', 'Credit', 'Churn', 'Diabetes', 'Magic']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    r_values = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    
    results = []
    print("=== MUST 3: RIGOROUS FIXED-GATE SELECTION ON VALIDATION ===")
    
    for d_name in datasets:
        try:
            _, (X_te_n_raw, X_te_c_raw, y_te_raw), card = load_openml_dataset(d_name)
        except: continue
        num_num = X_te_n_raw.shape[1]
        
        for seed in [42, 123, 2024]:
            set_seed(seed)
            # Tách Test Set hiện tại thành 50% Val, 50% Test
            xn_v, xn_t, xc_v, xc_t, y_v, y_t = train_test_split(X_te_n_raw, X_te_c_raw, y_te_raw, test_size=0.5, random_state=seed)
            
            # Corrupt 2 tập
            ev_n_v, ev_c_v, m_v = apply_synthetic_corruption(xn_v, xc_v, card, 0.2)
            ev_n_t, ev_c_t, m_t = apply_synthetic_corruption(xn_t, xc_t, card, 0.2)
            
            dl_val = DataLoader(PairedTabularDataset(xn_v, xc_v, ev_n_v, ev_c_v, y_v, m_v), batch_size=256)
            dl_test = DataLoader(PairedTabularDataset(xn_t, xc_t, ev_n_t, ev_c_t, y_t, m_t), batch_size=256)
            
            model = CART(num_num, card).to(device)
            ckpt = f"checkpoints/{EXP_VERSION}/seed{seed}/{d_name}_CART-v8-Dynamic.pt"
            if not os.path.exists(ckpt): continue
            model.load_state_dict(torch.load(ckpt, map_location=device))
            model.eval()
            
            def eval_sweep(loader):
                accs = {r: [] for r in r_values}
                acc_dyn, all_y = [], []
                with torch.no_grad():
                    for batch in loader:
                        xn_cr, xc_cr, y_batch = batch[2].to(device), batch[3].to(device), batch[4].to(device)
                        for r in r_values:
                            logits = get_sweep_logits(model, xn_cr, xc_cr, fixed_r=r)
                            accs[r].extend(torch.argmax(logits, dim=1).cpu().numpy())
                        logits_dyn = get_sweep_logits(model, xn_cr, xc_cr, fixed_r=None)
                        acc_dyn.extend(torch.argmax(logits_dyn, dim=1).cpu().numpy())
                        all_y.extend(y_batch.cpu().numpy())
                return {r: accuracy_score(all_y, accs[r])*100 for r in r_values}, accuracy_score(all_y, acc_dyn)*100
                
            # 1. Quét trên Val
            val_res, _ = eval_sweep(dl_val)
            best_r = max(val_res, key=val_res.get) # Chọn r*
            
            # 2. Đánh giá duy nhất r* và Dynamic trên Test
            test_res, test_dyn = eval_sweep(dl_test)
            val_selected_acc = test_res[best_r]
            
            results.append({
                'Dataset': d_name, 'Seed': seed,
                'Best_r_val': best_r,
                'ValSelected_Fixed': val_selected_acc,
                'Dynamic': test_dyn
            })
            print(f"[{d_name}-{seed}] Val Picked r={best_r} -> Test Fixed: {val_selected_acc:.1f} | Test Dynamic: {test_dyn:.1f}")

    df = pd.DataFrame(results).groupby('Dataset').mean().reset_index()
    df.to_csv("results/fixed_sweep_rigorous.csv", index=False)
    print("\nDone!")

if __name__ == "__main__":
    main()