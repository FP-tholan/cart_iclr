import torch
import torch.nn as nn
from models.cart import Tokenizer, ContextBlock

class TabAE_E2E(nn.Module):
    """ Denoising AutoEncoder dùng Transformer MÀ KHÔNG CÓ LOO """
    def __init__(self, num_numerical, cat_cardinalities, embed_dim=32, num_heads=4, num_classes=2):
        super().__init__()
        self.num_num = num_numerical
        self.num_cat = len(cat_cardinalities)
        
        self.tokenizer = Tokenizer(num_numerical, cat_cardinalities, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self.encoder = nn.ModuleList([ContextBlock(embed_dim, num_heads) for _ in range(2)])
        self.trust_net = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, 1))
        
        self.num_reconstruct = nn.Linear(embed_dim, 1) if num_numerical > 0 else None
        self.cat_reconstruct = nn.ModuleList([nn.Linear(embed_dim, card) for card in cat_cardinalities])
        
        self.classifier = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 64), nn.GELU(), nn.Linear(64, num_classes))

    def forward(self, xn, xc):
        batch_size = xn.shape[0]
        z = self.tokenizer(xn, xc)
        
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        c = torch.cat([cls_tokens, z], dim=1)
        for block in self.encoder: c = block(c)
            
        c_cls = c[:, 0, :]
        c_feat = c[:, 1:, :]
        
        logits = self.classifier(c_cls)
        trust_logits = self.trust_net(c_feat).squeeze(-1)
        r = torch.sigmoid(trust_logits)
        
        rec_num = self.num_reconstruct(c_feat[:, :self.num_num, :]).squeeze(-1) if self.num_num > 0 else None
        rec_cat = [self.cat_reconstruct[j](c_feat[:, self.num_num + j, :]) for j in range(self.num_cat)]
        
        return logits, trust_logits, r, None, rec_num, rec_cat, None

def train_tabae_e2e(model, loader, device, epochs=80):
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    ce_loss = nn.CrossEntropyLoss(reduction='none')
    model.train()
    for epoch in range(epochs):
        for xn_cl, xc_cl, xn_cr, xc_cr, y, mask in loader:
            xn_cr, xc_cr, y, mask = xn_cr.to(device), xc_cr.to(device), y.to(device), mask.to(device)
            xn_cl, xc_cl = xn_cl.to(device), xc_cl.to(device)
            
            logits, trust_logits, _, _, rec_num, rec_cat, _ = model(xn_cr, xc_cr)
            
            loss_cls = ce_loss(logits, y).mean()
            pos_weight = (mask == 0).sum() / ((mask == 1).sum() + 1e-8)
            loss_detect = nn.BCEWithLogitsLoss(pos_weight=pos_weight)(trust_logits, mask)
            
            loss_rec = 0.0
            if model.num_num > 0:
                loss_rec += torch.nn.functional.mse_loss(rec_num, xn_cl)
            for j in range(model.num_cat):
                loss_rec += ce_loss(rec_cat[j], xc_cl[:, j]).mean()
                
            loss = loss_cls + loss_detect + 4.0 * loss_rec
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()