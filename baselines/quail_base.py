import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class QuAILBaseline(nn.Module):
    """
    QuAIL: Quality-Aware Inertial Learning for Robust Training under Data Corruption.
    
    Dựa trên paper: https://arxiv.org/abs/2602.03686
    
    Ý tưởng chính:
    - Học một **feature reliability prior** cho từng đặc trưng (cột) dựa trên toàn bộ sample.
    - Sử dụng reliability để **điều chỉnh (modulate)** các embeddings trước khi phân loại.
    - Có thể tích hợp **proximal regularizer** (tùy chọn) để ổn định quá trình học.
    
    Khác biệt với CART:
    - QuAIL: Column-level reliability (1 giá trị cho cả cột), không có repair.
    - CART: Cell-level reliability (mỗi ô có trust riêng) + repair imputation.
    """
    def __init__(self, num_numerical, cat_cardinalities, embed_dim=32, num_classes=2):
        super().__init__()
        self.num_num = num_numerical
        self.num_cat = len(cat_cardinalities)
        self.num_features = num_numerical + self.num_cat
        self.embed_dim = embed_dim
        
        # ---------- 1. Feature Tokenizer (giống CART để so sánh công bằng) ----------
        if num_numerical > 0:
            self.num_weights = nn.Parameter(torch.Tensor(num_numerical, embed_dim))
            self.num_biases = nn.Parameter(torch.Tensor(num_numerical, embed_dim))
            nn.init.kaiming_uniform_(self.num_weights, a=math.sqrt(5))
            nn.init.zeros_(self.num_biases)
        
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(card, embed_dim) for card in cat_cardinalities
        ])
        
        self.feature_id = nn.Embedding(self.num_features, embed_dim)
        self.type_emb = nn.Embedding(2, embed_dim)  # 0: numerical, 1: categorical
        
        self.quality_context = nn.MultiheadAttention(
            embed_dim, num_heads=4, dropout=0.1, batch_first=True
        )
        self.quality_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, 1)  # reliability score cho mỗi feature
        )
        
        # ---------- 2. Classifier ----------
        # MLP 2 lớp, tương tự như các baseline khác
        self.classifier = nn.Sequential(
            nn.Linear(self.num_features * embed_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, num_classes)
        )
        
        # ---------- 3. Proximal Regularizer (tùy chọn) ----------
        # Trong paper, QuAIL sử dụng proximal term để hạn chế sự thay đổi của quality
        self.proximal_weight = 0.01 

    def forward(self, x_num, x_cat):
        batch_size = x_num.shape[0] if self.num_num > 0 else x_cat.shape[0]
        device = x_num.device if self.num_num > 0 else x_cat.device
        
        # ---------- Tokenization ----------
        tokens = []
        if self.num_num > 0:
            num_tokens = x_num.unsqueeze(-1) * self.num_weights.unsqueeze(0) + self.num_biases.unsqueeze(0)
            tokens.append(num_tokens)
        for j in range(self.num_cat):
            tokens.append(self.cat_embeddings[j](x_cat[:, j]).unsqueeze(1))
        
        z = torch.cat(tokens, dim=1)  # (B, F, D)
        
        f_ids = torch.arange(self.num_features, device=device).unsqueeze(0).expand(batch_size, -1)
        type_ids = torch.cat([
            torch.zeros(self.num_num, dtype=torch.long, device=device),
            torch.ones(self.num_cat, dtype=torch.long, device=device)
        ]).unsqueeze(0).expand(batch_size, -1)
        
        z = z + self.feature_id(f_ids) + self.type_emb(type_ids)
        
        # ---------- Quality Estimation ----------
        context, _ = self.quality_context(z, z, z)  # (B, F, D)
        
        reliability = torch.sigmoid(self.quality_net(context).squeeze(-1))  
        
        # ---------- Feature Modulation ----------
        # z_mod = reliability * z
        r_expanded = reliability.unsqueeze(-1)  # (B, F, 1)
        z_mod = z * r_expanded
        
        # ---------- Classification ----------
        z_flat = z_mod.view(batch_size, -1)
        logits = self.classifier(z_flat)
        
        return logits, reliability

    def get_proximal_loss(self, reliability_prev, reliability_curr):
        """
        Proximal regularizer: hạn chế sự thay đổi của reliability qua các bước.
        """
        return F.mse_loss(reliability_curr, reliability_prev)