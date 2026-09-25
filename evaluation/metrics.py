import numpy as np
from sklearn.metrics import auc, roc_auc_score, average_precision_score

def calc_relative_degradation(acc_clean, acc_corr):
    """Điểm 24: Relative Degradation (RD). Càng thấp càng tốt."""
    return (acc_clean - acc_corr) / acc_clean * 100

def calc_r_auc(accuracies, severities=[0.0, 0.1, 0.2, 0.3]):
    """Điểm 25: Robustness AUC (R-AUC). Tích phân Accuracy dọc theo các mức độ nhiễu."""
    # Sắp xếp lại cho chắc chắn trục X tăng dần
    sorted_idx = np.argsort(severities)
    x = np.array(severities)[sorted_idx]
    y = np.array(accuracies)[sorted_idx]
    return auc(x, y)

def calc_cell_detection(trust_scores, true_masks):
    """
    Điểm 26: Cell Detection AUROC & AUPRC.
    trust_scores: r_ij dự đoán từ CART
    true_masks: c_ij (1 = Clean, 0 = Corrupted)
    """
    # Làm phẳng ma trận để tính AUC cho toàn bộ các ô
    y_true = true_masks.flatten()
    y_score = trust_scores.flatten()
    
    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)
    return auroc, auprc

def calc_trust_calibration(trust_scores, true_masks, bins=10):
    """Điểm 27: Trust Calibration. Kiểm tra xem điểm r_ij có phản ánh đúng xác suất ô đó sạch không."""
    y_true = true_masks.flatten()
    y_score = trust_scores.flatten()
    
    bucket_limits = np.linspace(0, 1, bins + 1)
    calibration_data = []
    
    for i in range(bins):
        low, high = bucket_limits[i], bucket_limits[i+1]
        mask = (y_score >= low) & (y_score <= high)
        if np.sum(mask) > 0:
            true_clean_ratio = np.mean(y_true[mask])
            avg_trust = np.mean(y_score[mask])
            calibration_data.append((avg_trust, true_clean_ratio))
            
    return calibration_data