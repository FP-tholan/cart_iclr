import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from evaluation.metrics import calc_cell_detection

class DAE(nn.Module):
    def __init__(self, num_numerical, cat_cardinalities, embed_dim=32, hidden_dim=128):
        super().__init__()
        self.num_num = num_numerical
        self.num_cat = len(cat_cardinalities)
        self.cat_embs = nn.ModuleList([nn.Embedding(c, embed_dim) for c in cat_cardinalities])
        input_dim = num_numerical + len(cat_cardinalities) * embed_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.num_head = nn.Linear(hidden_dim, num_numerical) if num_numerical > 0 else None
        self.cat_heads = nn.ModuleList([nn.Linear(hidden_dim, c) for c in cat_cardinalities])
        
    def forward(self, xn, xc):
        embs = [xn] if self.num_num > 0 else []
        for j in range(self.num_cat):
            embs.append(self.cat_embs[j](xc[:, j]))
        h = torch.cat(embs, dim=1)
        feat = self.net(h)
        rec_num = self.num_head(feat) if self.num_num > 0 else None
        rec_cat = [head(feat) for head in self.cat_heads]
        return rec_num, rec_cat

def train_dae(model, loader, device, epochs=40):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ce_loss = nn.CrossEntropyLoss(reduction='mean')
    model.train()
    for epoch in range(epochs):
        for xn_cl, xc_cl, xn_cr, xc_cr, _, _ in loader:
            xn_cl, xc_cl = xn_cl.to(device), xc_cl.to(device)
            xn_cr, xc_cr = xn_cr.to(device), xc_cr.to(device)
            rec_num, rec_cat = model(xn_cr, xc_cr)
            loss = 0.0
            if model.num_num > 0: loss += F.mse_loss(rec_num, xn_cl)
            for j in range(model.num_cat): loss += ce_loss(rec_cat[j], xc_cl[:, j])
            opt.zero_grad()
            loss.backward()
            opt.step()

def extract_dae_repaired_data(model, loader, device):
    model.eval()
    X_n_rep, X_c_rep, Y_all = [], [], []
    with torch.no_grad():
        for batch in loader:
            xn_cr, xc_cr, y = batch[2].to(device), batch[3].to(device), batch[4]
            rec_num, rec_cat = model(xn_cr, xc_cr)
            X_n_rep.append(rec_num.cpu().numpy() if model.num_num > 0 else np.empty((xn_cr.shape[0], 0)))
            xc_rep_list = []
            for j in range(model.num_cat):
                xc_rep_list.append(torch.argmax(rec_cat[j], dim=-1).cpu().numpy())
            X_c_rep.append(np.column_stack(xc_rep_list) if xc_rep_list else np.empty((xn_cr.shape[0], 0)))
            Y_all.append(y.numpy())
    return np.vstack(X_n_rep), np.vstack(X_c_rep), np.concatenate(Y_all)

def eval_repair_metrics_dae(model, loader, device):
    model.eval()
    all_trust, all_masks = [], []
    num_mae_sum, num_corrupt_cnt, cat_correct, cat_corrupt_cnt = 0.0, 0.0, 0.0, 0.0
    with torch.no_grad():
        for batch in loader:
            xn_cl, xc_cl, xn_cr, xc_cr, _, m = batch
            xn_cl, xc_cl = xn_cl.to(device), xc_cl.to(device)
            xn_cr, xc_cr = xn_cr.to(device), xc_cr.to(device)
            rec_num, rec_cat = model(xn_cr, xc_cr)
            trust_list = []
            if model.num_num > 0:
                err = torch.abs(rec_num - xn_cr)
                trust_list.append(torch.exp(-err))
            for j in range(model.num_cat):
                probs = F.softmax(rec_cat[j], dim=-1)
                prob_obs = probs.gather(1, xc_cr[:, j].unsqueeze(1)).squeeze(1)
                trust_list.append(prob_obs.unsqueeze(1))
            all_trust.append(torch.cat(trust_list, dim=1).cpu().numpy())
            all_masks.append(m.numpy())
            m = m.to(device)
            inv_mask = 1.0 - m
            if model.num_num > 0:
                err_cl = torch.abs(rec_num - xn_cl)
                num_mae_sum += (err_cl * inv_mask[:, :model.num_num]).sum().item()
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