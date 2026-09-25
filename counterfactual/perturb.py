import torch

def generate_single_cell_counterfactual(xn, xc, cardinalities, feature_idx):
    """
    Tạo Counterfactual bằng cách nhiễu (perturb) duy nhất 1 ô dữ liệu (feature_idx).
    - Numerical: Cộng nhiễu Gaussian nhỏ.
    - Categorical: Thay thế bằng một hạng mục (category) khác.
    """
    xn_cf = xn.clone()
    xc_cf = xc.clone()
    num_num = xn.shape[1]
    
    if feature_idx < num_num:
        # Xử lý biến liên tục (Numerical)
        std_val = xn[:, feature_idx].std() + 1e-6
        # Bounded perturbation: epsilon = 1.0 độ lệch chuẩn
        noise = torch.randn_like(xn[:, feature_idx]) * std_val
        xn_cf[:, feature_idx] = xn_cf[:, feature_idx] + noise
    else:
        # Xử lý biến phân loại (Categorical)
        cat_idx = feature_idx - num_num
        card = cardinalities[cat_idx]
        if card > 1:
            # Chọn ngẫu nhiên một category khác
            rand_cats = torch.randint(0, card, (xc.shape[0],), device=xc.device)
            xc_cf[:, cat_idx] = rand_cats
            
    return xn_cf, xc_cf