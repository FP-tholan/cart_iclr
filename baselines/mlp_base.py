import torch
import torch.nn as nn

class VanillaMLP(nn.Module):
    def __init__(self, num_numerical, cat_cardinalities, embed_dim=16, hidden_dim=128, num_classes=2):
        super().__init__()
        self.num_numerical = num_numerical
        self.num_categorical = len(cat_cardinalities)
        
        self.num_encoders = nn.ModuleList([nn.Linear(1, embed_dim) for _ in range(num_numerical)])
        self.cat_encoders = nn.ModuleList([nn.Embedding(c, embed_dim) for c in cat_cardinalities])
        
        in_features = (num_numerical + self.num_categorical) * embed_dim
        self.predictor = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x_num, x_cat):
        z_list = []
        for j in range(self.num_numerical):
            z_list.append(self.num_encoders[j](x_num[:, j:j+1]))
        for j in range(self.num_categorical):
            z_list.append(self.cat_encoders[j](x_cat[:, j]))
            
        z = torch.stack(z_list, dim=1)
        z_flat = z.view(x_num.shape[0], -1)
        return self.predictor(z_flat)