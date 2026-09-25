import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE_E2E(nn.Module):
    """ Variational AutoEncoder tiêu chuẩn cho Tabular Data """
    def __init__(self, num_numerical, cat_cardinalities, embed_dim=32, hidden_dim=128, num_classes=2):
        super().__init__()
        self.num_num = num_numerical
        self.num_cat = len(cat_cardinalities)
        self.cardinalities = cat_cardinalities
        self.dim = num_numerical + sum(cat_cardinalities) 
        
        # Encoder
        self.enc1 = nn.Linear(self.dim, hidden_dim)
        self.enc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, embed_dim)
        self.logvar = nn.Linear(hidden_dim, embed_dim)
        
        # Decoder
        self.dec1 = nn.Linear(embed_dim, hidden_dim)
        self.dec2 = nn.Linear(hidden_dim, self.dim)
        
        # Classifier (Attached to Latent Space)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, num_classes)
        )

    def encode(self, x):
        h = F.relu(self.enc1(x))
        h = F.relu(self.enc2(h))
        return self.mu(h), self.logvar(h)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, xn, xc):
        # One-hot encode Categorical
        x_list = [xn] if self.num_num > 0 else []
        for j, c in enumerate(self.cardinalities):
            x_list.append(F.one_hot(xc[:, j], num_classes=c).float())
        x = torch.cat(x_list, dim=1)
        
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = F.relu(self.dec1(z))
        x_hat = self.dec2(x_hat)
        
        # Split Reconstruction
        rec_num = x_hat[:, :self.num_num] if self.num_num > 0 else None
        rec_cat = []
        idx = self.num_num
        for c in self.cardinalities:
            rec_cat.append(x_hat[:, idx:idx+c])
            idx += c
            
        # Trust Score: Định nghĩa = Nghịch đảo của Reconstruction Error (Lỗi càng thấp -> Trust càng cao)
        trust_logits_list = []
        if self.num_num > 0:
            err_num = (xn - rec_num)**2
            trust_logits_list.append(-err_num) 
        for j, c in enumerate(self.cardinalities):
            prob = F.softmax(rec_cat[j], dim=-1)
            obs_prob = prob.gather(1, xc[:, j].unsqueeze(1)).squeeze(1)
            logit_cat = torch.log(obs_prob / (1.0 - obs_prob + 1e-8))
            trust_logits_list.append(logit_cat.unsqueeze(1))
            
        trust_logits = torch.cat(trust_logits_list, dim=1)
        r = torch.sigmoid(trust_logits)
        
        logits = self.classifier(mu)
        return logits, trust_logits, r, None, rec_num, rec_cat, logvar

def train_vae_e2e(model, loader, device, epochs=80):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ce_loss = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        for xn_cl, xc_cl, xn_cr, xc_cr, y, _ in loader:
            xn_cr, xc_cr, y = xn_cr.to(device), xc_cr.to(device), y.to(device)
            xn_cl, xc_cl = xn_cl.to(device), xc_cl.to(device)
            
            logits, _, _, _, rec_num, rec_cat, logvar = model(xn_cr, xc_cr)
            
            loss_cls = ce_loss(logits, y)
            loss_rec = 0.0
            if model.num_num > 0:
                loss_rec += F.mse_loss(rec_num, xn_cl)
            for j in range(model.num_cat):
                loss_rec += ce_loss(rec_cat[j], xc_cl[:, j])
                
            x_cr_cat = torch.cat([xn_cr] + [F.one_hot(xc_cr[:,j], model.cardinalities[j]).float() for j in range(model.num_cat)], dim=1)
            mu, logvar = model.encode(x_cr_cat)
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / xn_cr.size(0)
            
            loss = loss_cls + loss_rec + 0.1 * kld_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()