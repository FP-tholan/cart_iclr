import torch
import torch.nn as nn
import torch.nn.functional as F

class FTTransformer(nn.Module):
    def __init__(self, num_numerical, cat_cardinalities, embed_dim=16, depth=3, heads=4, num_classes=2):
        super().__init__()
        self.num_numerical = num_numerical
        self.num_categorical = len(cat_cardinalities)
        self.num_features = num_numerical + self.num_categorical
        
        # 1. Feature Tokenizer
        self.num_encoders = nn.ModuleList([nn.Linear(1, embed_dim) for _ in range(num_numerical)])
        self.cat_encoders = nn.ModuleList([nn.Embedding(c, embed_dim) for c in cat_cardinalities])
        
        # 2. CLS Token 
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # 3. Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=heads, 
            dim_feedforward=embed_dim * 4, 
            dropout=0.1, 
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # 4. Prediction Head 
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, num_classes)
        )

    def forward(self, x_num, x_cat):
        batch_size = x_num.shape[0]
        
        # --- A. Tokenization ---
        z_list = []
        for j in range(self.num_numerical):
            z_list.append(self.num_encoders[j](x_num[:, j:j+1]))
        for j in range(self.num_categorical):
            z_list.append(self.cat_encoders[j](x_cat[:, j]))
            
        z = torch.stack(z_list, dim=1) 
        
        # --- B. Add CLS Token ---
        cls_tokens = self.cls_token.expand(batch_size, -1, -1) 
        z = torch.cat([cls_tokens, z], dim=1) \
        
        # --- C. Transformer & Prediction ---
        z_out = self.transformer(z)
        cls_out = z_out[:, 0, :] 
        
        return self.head(cls_out)