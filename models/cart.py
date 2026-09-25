import torch
import torch.nn as nn
import math

class Tokenizer(nn.Module):
    def __init__(self, num_numerical, cat_cardinalities, embed_dim):
        super().__init__()
        self.num_num = num_numerical
        self.num_cat = len(cat_cardinalities)
        self.total_features = self.num_num + self.num_cat
        
        if self.num_num > 0:
            self.num_weights = nn.Parameter(torch.Tensor(self.num_num, embed_dim))
            self.num_biases = nn.Parameter(torch.Tensor(self.num_num, embed_dim))
            nn.init.kaiming_uniform_(self.num_weights, a=math.sqrt(5))
            nn.init.zeros_(self.num_biases)
            
        self.cat_embeddings = nn.ModuleList([nn.Embedding(card, embed_dim) for card in cat_cardinalities])
        self.feature_id = nn.Embedding(self.total_features, embed_dim)
        self.type_emb = nn.Embedding(2, embed_dim)

    def forward(self, x_num, x_cat):
        batch_size = x_num.shape[0] if self.num_num > 0 else x_cat.shape[0]
        device = x_num.device if self.num_num > 0 else x_cat.device
        
        tokens = []
        if self.num_num > 0:
            tokens.append(x_num.unsqueeze(-1) * self.num_weights.unsqueeze(0) + self.num_biases.unsqueeze(0))
        for j in range(self.num_cat):
            tokens.append(self.cat_embeddings[j](x_cat[:, j]).unsqueeze(1))
            
        z = torch.cat(tokens, dim=1) if tokens else torch.empty(batch_size, 0, self.num_weights.size(1), device=device)
        
        f_ids = torch.arange(self.total_features, device=device).unsqueeze(0).expand(batch_size, -1)
        z = z + self.feature_id(f_ids)
        type_ids = torch.cat([torch.zeros(self.num_num, dtype=torch.long, device=device), 
                              torch.ones(self.num_cat, dtype=torch.long, device=device)])
        z = z + self.type_emb(type_ids.unsqueeze(0).expand(batch_size, -1))
        return z

class ContextBlock(nn.Module):
    """Transformer block thông thường dùng cho nhánh Trust và Prediction"""
    def __init__(self, dim, heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim))

    def forward(self, x, attn_mask=None):
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x

class LOOContextBlock(nn.Module):
    """
    SỬA LỖI P0: KHỐI LOO THỰC SỰ (TRUE LEAVE-ONE-OUT)
    Query độc lập hoàn toàn với giá trị quan sát của ô dữ liệu.
    """
    def __init__(self, dim, heads, dropout=0.1):
        super().__init__()
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_out = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim))

    def forward(self, z, query_tokens):
        B, D, _ = z.shape
        # Query độc lập: chỉ là vector định danh đặc trưng (Feature Identity)
        q = query_tokens.expand(B, -1, -1) 
        kv = self.norm_kv(z)

        # Mặt nạ cấm nhìn chính mình (True LOO)
        loo_mask = torch.eye(D, device=z.device, dtype=torch.bool)

        # Chú ý: Đầu ra attn_out chỉ được cộng residual với q (không chứa z)
        attn_out, _ = self.attn(q, kv, kv, attn_mask=loo_mask)
        ctx = q + attn_out 
        ctx = ctx + self.ffn(self.norm_out(ctx))
        return ctx

class CART(nn.Module):
    def __init__(self, num_numerical, cat_cardinalities, embed_dim=32, num_heads=4, num_classes=2):
        super().__init__()
        self.num_num = num_numerical
        self.num_cat = len(cat_cardinalities)
        self.total_features = num_numerical + len(cat_cardinalities)
        self.embed_dim = embed_dim
        
        self.tokenizer = Tokenizer(num_numerical, cat_cardinalities, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 1. Trust Branch: Full Context
        self.trust_context = nn.ModuleList([ContextBlock(embed_dim, num_heads) for _ in range(2)])
        self.trust_net = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.GELU(), nn.Linear(embed_dim, 1))
        
        # 2. Repair Branch: Đã sửa thành LOOContextBlock
        self.repair_context = LOOContextBlock(embed_dim, num_heads)
        self.repair_net = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim))
        
        # 3. Explicit Reconstruction
        self.num_reconstruct = nn.Linear(embed_dim, 1) if num_numerical > 0 else None
        self.cat_reconstruct = nn.ModuleList([nn.Linear(embed_dim, card) for card in cat_cardinalities])
        
        # 4. Prediction Encoder
        self.pred_encoder = nn.ModuleList([ContextBlock(embed_dim, num_heads) for _ in range(2)])
        self.classifier = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 64), nn.GELU(), nn.Linear(64, num_classes))

    def forward(self, x_num, x_cat):
        batch_size = x_num.shape[0] if self.num_num > 0 else x_cat.shape[0]
        device = x_num.device if self.num_num > 0 else x_cat.device
        
        z = self.tokenizer(x_num, x_cat)
        
        # --- Trust Branch ---
        c_trust = z
        for block in self.trust_context: 
            c_trust = block(c_trust)
        trust_logits = self.trust_net(torch.cat([z, c_trust], dim=-1)).squeeze(-1)
        r = torch.sigmoid(trust_logits)
        
        # --- Repair Branch (True LOO SẠCH) ---
        f_ids = torch.arange(self.total_features, device=device).unsqueeze(0).expand(batch_size, -1)
        f_embs = self.tokenizer.feature_id(f_ids)
        
        # Đưa f_embs làm query, mô hình tự lục lọi z (nhưng bị mask chéo) để tái tạo
        c_repair = self.repair_context(z, query_tokens=f_embs)
        z_repaired = self.repair_net(torch.cat([c_repair, f_embs], dim=-1))
        
        # --- Adaptive Fusion ---
        r_expanded = r.unsqueeze(-1)
        z_star = r_expanded * z + (1.0 - r_expanded) * z_repaired
        
        # --- Reconstruction Targets ---
        rec_num = self.num_reconstruct(z_repaired[:, :self.num_num, :]).squeeze(-1) if self.num_num > 0 else None
        rec_cat = [self.cat_reconstruct[j](z_repaired[:, self.num_num + j, :]) for j in range(self.num_cat)]
        
        # --- Prediction ---
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        c_star = torch.cat([cls_tokens, z_star], dim=1)
        for block in self.pred_encoder: 
            c_star = block(c_star)
            
        logits = self.classifier(c_star[:, 0, :])
        return logits, trust_logits, r, z_star, rec_num, rec_cat, z_repaired