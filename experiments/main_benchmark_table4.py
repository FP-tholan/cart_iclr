import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from functools import partial
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, auc
import pandas as pd
import numpy as np
import random
import warnings
import copy
warnings.filterwarnings("ignore")
import joblib

from models.cart import CART
from baselines.quail_base import QuAILBaseline
from baselines.mlp_base import VanillaMLP
from baselines.ft_transformer import FTTransformer
from baselines.xgboost_base import XGBoostBaseline
from baselines.catboost_base import CatBoostBaseline
from data.loaders import load_openml_dataset, apply_synthetic_corruption
from evaluation.metrics import calc_relative_degradation, calc_cell_detection

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calc_nr_auc(accuracies, severities):
    return auc(severities, accuracies) / (severities[-1] - severities[0])

def update_ema_variables(model, ema_model, alpha=0.995):
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(param.data, alpha=1 - alpha)

class CleanTabularDataset(Dataset):
    def __init__(self, X_num, X_cat, y):
        self.X_num = torch.tensor(X_num, dtype=torch.float32)
        self.X_cat = torch.tensor(X_cat, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X_num[idx], self.X_cat[idx], self.y[idx]

class PairedTabularDataset(Dataset):
    def __init__(self, x_n_cl, x_c_cl, x_n_cr, x_c_cr, y, mask):
        self.x_n_cl = torch.tensor(x_n_cl, dtype=torch.float32)
        self.x_c_cl = torch.tensor(x_c_cl, dtype=torch.long)
        self.x_n_cr = torch.tensor(x_n_cr, dtype=torch.float32)
        self.x_c_cr = torch.tensor(x_c_cr, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)
        self.mask = torch.tensor(mask, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        return self.x_n_cl[idx], self.x_c_cl[idx], self.x_n_cr[idx], self.x_c_cr[idx], self.y[idx], self.mask[idx]

def dynamic_corrupt_collate(batch, card):
    xn_cl = torch.stack([b[0] for b in batch])
    xc_cl = torch.stack([b[1] for b in batch])
    y = torch.stack([b[2] for b in batch])
    
    rate = random.uniform(0.05, 0.30)
    xn_cr, mask_num = xn_cl.clone(), torch.ones_like(xn_cl)
    if xn_cl.shape[1] > 0:
        rand_mask_n = (torch.rand_like(xn_cl) < rate).float()
        noise = torch.randn_like(xn_cl) * 2.0
        xn_cr = xn_cl + noise * rand_mask_n
        mask_num = 1.0 - rand_mask_n

    xc_cr, mask_cat = xc_cl.clone(), torch.ones_like(xc_cl, dtype=torch.float32)
    if xc_cl.shape[1] > 0:
        rand_mask_c = (torch.rand_like(xc_cl.float()) < rate)
        effective_mask = torch.zeros_like(rand_mask_c) # Sửa theo yêu cầu của Reviewer
        
        for j, c in enumerate(card):
            if c > 1: # Không corrupt nếu cardinality <= 1
                effective_mask[:, j] = rand_mask_c[:, j]
                orig = xc_cl[:, j]
                random_cat = torch.randint(0, c, (len(batch),), device=xc_cl.device)
                same = (random_cat == orig)
                # Nếu trùng, ép nhảy sang category kế tiếp
                random_cat = torch.where(same, (random_cat + 1) % c, random_cat)
                xc_cr[effective_mask[:, j], j] = random_cat[effective_mask[:, j]]
                
        mask_cat = 1.0 - effective_mask.float()

    mask = torch.cat([mask_num, mask_cat], dim=1)
    return xn_cl, xc_cl, xn_cr, xc_cr, y, mask

def train_cart_v8(model, loader, device, epochs=80):
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    ce_loss = nn.CrossEntropyLoss(reduction='none')
    ema_teacher = None 

    switch_epoch = int(epochs * 30 / 80) 
    
    model.train()
    for epoch in range(epochs):
        is_stage_1 = epoch < switch_epoch
        if epoch == switch_epoch and ema_teacher is None:
            ema_teacher = copy.deepcopy(model)
            ema_teacher.eval()
            for param in ema_teacher.parameters(): param.requires_grad = False
            
        for xn_cl, xc_cl, xn_cr, xc_cr, y, mask in loader:
            xn_cr, xc_cr, y, mask = xn_cr.to(device), xc_cr.to(device), y.to(device), mask.to(device)
            xn_cl, xc_cl = xn_cl.to(device), xc_cl.to(device)
            
            logits_corr, trust_logits, r, z_star, rec_num, rec_cat, _ = model(xn_cr, xc_cr)
            
            loss_cls = ce_loss(logits_corr, y).mean()
            pos_weight = (mask == 0).sum() / ((mask == 1).sum() + 1e-8)
            loss_detect = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(trust_logits, mask)
            
            loss_rec = 0.0
            inv_mask = 1.0 - mask
            if model.num_num > 0:
                loss_rec += (F.mse_loss(rec_num, xn_cl, reduction='none') * inv_mask[:, :model.num_num]).sum() / (inv_mask[:, :model.num_num].sum() + 1e-8)
            for j in range(model.num_cat):
                loss_rec += (ce_loss(rec_cat[j], xc_cl[:, j]) * inv_mask[:, model.num_num + j]).sum() / (inv_mask[:, model.num_num + j].sum() + 1e-8)
            
            if is_stage_1:
                loss = 1.0 * loss_detect + 4.0 * loss_rec
            else:
                with torch.no_grad():
                    logits_clean, _, _, _, _, _, _ = ema_teacher(xn_cl, xc_cl)
                    p_clean = F.softmax(logits_clean, dim=-1)
                    teacher_conf = p_clean.max(dim=1).values
                
                conf_weight = torch.clamp((teacher_conf - 0.5) / 0.5, min=0.0)
                kl = F.kl_div(F.log_softmax(logits_corr, dim=-1), p_clean, reduction='none').sum(dim=-1)
                loss_cons = (conf_weight * kl).mean()
                loss = 1.0 * loss_cls + 1.0 * loss_detect + 4.0 * loss_rec + 0.5 * loss_cons
                
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            if not is_stage_1 and ema_teacher is not None:
                update_ema_variables(model, ema_teacher)

def eval_model(model, loader, device, model_type="nn", X_eval=None, y_eval=None):
    all_preds, all_probs, all_y = [], [], []
    if model_type == "nn":
        model.eval()
        with torch.no_grad():
            for batch in loader:
                xn_cl, xc_cl, xn_cr, xc_cr, y, mask = batch
                xn, xc, y_tensor = xn_cr.to(device), xc_cr.to(device), y.to(device)
                out = model(xn, xc)
                logits = out[0] if isinstance(out, tuple) else out
                all_probs.extend(F.softmax(logits, dim=1)[:, 1].cpu().numpy())
                all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
                all_y.extend(y_tensor.cpu().numpy())
    else:
        all_y = y_eval
        all_preds = model.model.predict(X_eval)
        all_probs = model.model.predict_proba(X_eval)[:, 1]

    try: roc = roc_auc_score(all_y, all_probs) * 100
    except ValueError: roc = 50.0
    return {'acc': accuracy_score(all_y, all_preds) * 100, 'roc': roc}

def eval_repair_metrics(model, loader, device):
    model.eval()
    all_trust, all_masks = [], []
    num_mae_sum, num_corrupt_cnt, cat_correct, cat_corrupt_cnt = 0.0, 0.0, 0.0, 0.0
    
    from evaluation.metrics import calc_cell_detection
    
    with torch.no_grad():
        for batch in loader:
            xn_cl, xc_cl, xn_cr, xc_cr, _, m = batch
            xn_cl, xc_cl = xn_cl.to(device), xc_cl.to(device)
            _, _, r_score, _, rec_num, rec_cat, _ = model(xn_cr.to(device), xc_cr.to(device))
            
            all_trust.append(r_score.cpu().numpy())
            all_masks.append(m.numpy())
            
            m = m.to(device)
            inv_mask = 1.0 - m
            
            if model.num_num > 0 and rec_num is not None:
                err = torch.abs(rec_num - xn_cl)
                num_mae_sum += (err * inv_mask[:, :model.num_num]).sum().item()
                num_corrupt_cnt += inv_mask[:, :model.num_num].sum().item()
                
            for j in range(model.num_cat):
                preds_cat = torch.argmax(rec_cat[j], dim=-1)
                correct = (preds_cat == xc_cl[:, j]).float()
                cat_correct += (correct * inv_mask[:, model.num_num + j]).sum().item()
                cat_corrupt_cnt += inv_mask[:, model.num_num + j].sum().item()
                
    auroc, auprc = calc_cell_detection(np.vstack(all_trust), np.vstack(all_masks))
    num_mae = (num_mae_sum/num_corrupt_cnt) if num_corrupt_cnt > 0 else np.nan
    cat_acc = (cat_correct/cat_corrupt_cnt*100) if cat_corrupt_cnt > 0 else np.nan
    return auroc, auprc, num_mae, cat_acc

def main():
    EXP_VERSION = "main_results_table4"
    datasets = ['Adult', 'HELOC', 'Higgs', 'Bank', 'Credit', 'Churn', 'Diabetes', 'Magic']
    seeds = [42, 123, 2024]
    severities = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print(f"=== BENCHMARK ===")
    results = []

    for d_name in datasets:
        print(f"\n[{d_name}] Đang tải dữ liệu...")
        try:
            (X_tr_n_raw, X_tr_c_raw, y_tr_raw), (X_te_n_raw, X_te_c_raw, y_te_raw), card = load_openml_dataset(d_name)
            if len(set(y_tr_raw)) != 2: continue
        except Exception: continue
        num_num = X_tr_n_raw.shape[1]
        
        for seed in seeds:
            set_seed(seed)
            
            test_corruptions = {}
            for rate in severities:
                if rate == 0.0:
                    mask_ev = np.ones((X_te_n_raw.shape[0], num_num + len(card)), dtype=np.float32)
                    test_corruptions[rate] = (X_te_n_raw, X_te_c_raw, mask_ev)
                else:
                    test_corruptions[rate] = apply_synthetic_corruption(X_te_n_raw, X_te_c_raw, card, rate)

            # Dataloader động cho CART-Dynamic
            ds_clean = CleanTabularDataset(X_tr_n_raw, X_tr_c_raw, y_tr_raw)
            dl_train_dynamic = DataLoader(
                ds_clean, 
                batch_size=128, 
                shuffle=True, 
                collate_fn=partial(dynamic_corrupt_collate, card=card),
                num_workers=4,        
                pin_memory=True,       
                persistent_workers=True 
            )
            
            # Nhiễu cố định 20% cho Baselines VÀ CART-Fixed
            X_tr_n_corr, X_tr_c_corr, mask_tr = apply_synthetic_corruption(X_tr_n_raw, X_tr_c_raw, card, 0.2)
            dl_train_fixed = DataLoader(
                PairedTabularDataset(X_tr_n_raw, X_tr_c_raw, X_tr_n_corr, X_tr_c_corr, y_tr_raw, mask_tr), 
                batch_size=128, 
                shuffle=True,
                num_workers=4,          
                pin_memory=True,        
                persistent_workers=True 
            )

            models = {
                'MLP-Aug-80ep': (VanillaMLP(num_num, card), 'nn_base_80'),
                'FT-Transformer-Aug-80ep': (FTTransformer(num_num, card), 'nn_base_80'),
                'XGBoost-Aug': (XGBoostBaseline(), 'tree'),
                'QuAIL-Aug-80ep': (QuAILBaseline(num_num, card), 'nn_base_80'),
                'CatBoost-Aug': (CatBoostBaseline(), 'tree'),
                'CART-v8-Fixed20': (CART(num_num, card), 'nn_cart_fixed'),
                'CART-v8-Dynamic': (CART(num_num, card), 'nn_cart_dynamic'),
                'CART-v8-Dyn-30ep': (CART(num_num, card), 'nn_cart_dynamic_30')
            }

            ckpt_dir = f"checkpoints/{EXP_VERSION}/seed{seed}"
            os.makedirs(ckpt_dir, exist_ok=True)
            
            for m_name, (m_obj, m_type) in models.items():
                safe_m_name = m_name.replace(' ', '_').replace('(', '').replace(')', '')
                ckpt_path = f"{ckpt_dir}/{d_name}_{safe_m_name}.pt"
                
                if os.path.exists(ckpt_path):
                    if 'nn' in m_type:
                        m_obj.load_state_dict(torch.load(ckpt_path, map_location=device))
                        m_obj.to(device)
                    else: m_obj.model = joblib.load(ckpt_path)
                else:
                    if 'nn' in m_type:
                        m_obj.to(device)
                        if m_type == 'nn_cart_dynamic':
                            train_cart_v8(m_obj, dl_train_dynamic, device, epochs=80)
                        elif m_type == 'nn_cart_fixed':
                            train_cart_v8(m_obj, dl_train_fixed, device, epochs=80)
                        elif m_type == 'nn_cart_dynamic_30':  
                            train_cart_v8(m_obj, dl_train_dynamic, device, epochs=30) 
                        elif m_type in ['nn_base', 'nn_base_80']:
                            # Nếu là nn_base_80 thì chạy 80 epochs, ngược lại chạy 30
                            epochs_to_run = 80 if m_type == 'nn_base_80' else 30
                            opt = torch.optim.AdamW(m_obj.parameters(), lr=3e-4, weight_decay=1e-4)
                            cel = nn.CrossEntropyLoss()
                            m_obj.train()
                            for _ in range(epochs_to_run):
                                for batch in dl_train_fixed:
                                    xn, xc, y_b = batch[2].to(device), batch[3].to(device), batch[4].to(device)
                                    out = m_obj(xn, xc)
                                    logits = out[0] if isinstance(out, tuple) else out
                                    loss = cel(logits, y_b)
                                    
                                    opt.zero_grad()
                                    loss.backward()
                                    opt.step()
                            torch.save(m_obj.state_dict(), ckpt_path)
                    else:
                        m_obj.fit(X_tr_n_corr, X_tr_c_corr, y_tr_raw)
                        joblib.dump(m_obj.model, ckpt_path)

            for m_name, (m_obj, m_type) in models.items():
                acc_list = []
                m_20, m_0 = {}, {}
                c_auroc, c_auprc, rep_mae, rep_acc = np.nan, np.nan, np.nan, np.nan
                
                for rate in severities:
                    X_ev_n, X_ev_c, mask_ev = test_corruptions[rate]
                    dl_eval_paired = DataLoader(PairedTabularDataset(X_te_n_raw, X_te_c_raw, X_ev_n, X_ev_c, y_te_raw, mask_ev), batch_size=256)
                    
                    if m_type == 'tree':
                        X_eval = pd.concat([pd.DataFrame(X_ev_n, dtype=float), pd.DataFrame(X_ev_c, dtype=int)], axis=1) if len(card)>0 else pd.DataFrame(X_ev_n, dtype=float)
                        mets = eval_model(m_obj, dl_eval_paired, device, "tree", X_eval.astype(str).values if len(card)>0 else X_eval, y_te_raw)
                    else:
                        mets = eval_model(m_obj, dl_eval_paired, device, "nn")
                        
                    acc_list.append(mets['acc'])
                    if rate == 0.0: m_0 = mets
                    if rate == 0.2: 
                        m_20 = mets
                        if 'CART' in m_name:
                            c_auroc, c_auprc, rep_mae, rep_acc = eval_repair_metrics(m_obj, dl_eval_paired, device)

                res_dict = {
                    'Dataset': d_name, 'Seed': seed, 'Model': m_name,
                    'Clean_Acc': m_0['acc'], 'Acc_20': m_20['acc'],
                    'NR_AUC': calc_nr_auc(acc_list, severities),
                    'Cell_AUROC': c_auroc, 'Cell_AUPRC': c_auprc,
                    'Num_Rep_MAE': rep_mae, 'Cat_Rep_Acc': rep_acc
                }
                for idx, rate in enumerate(severities):
                    res_dict[f'Acc_{int(rate*100)}'] = acc_list[idx]
                results.append(res_dict)
            
    df = pd.DataFrame(results)
    
    metric_cols = ['Clean_Acc', 'Acc_20', 'NR_AUC', 'Cell_AUROC', 'Cell_AUPRC', 'Num_Rep_MAE', 'Cat_Rep_Acc']
    df_agg = df.groupby(['Dataset', 'Model'])[metric_cols].agg(['mean', 'std']).reset_index()
    
    df.to_csv("results/main_results_table4_raw.csv", index=False)
    df_agg.to_csv("results/main_results_table4_agg.csv", index=False)
    print("\nDone!")

if __name__ == "__main__":
    main()