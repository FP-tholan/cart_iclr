import pandas as pd
import numpy as np
from scipy.stats import wilcoxon
import scipy.stats as st

def main():
    try:
        df = pd.read_csv("results/ablation_7way_final_raw.csv")
    except FileNotFoundError:
        print("Không tìm thấy file kết quả.")
        return

    metric_cols = ['Raw', 'Fixed_80_20', 'Permuted_Trust', 'Dynamic_CART']
    df_agg = df.groupby('Dataset')[metric_cols].mean().reset_index()
    
    print("=== DATASET-LEVEL STATISTICAL TEST (n = 8) ===")
    
    def run_test(col_test, col_base, name):
        diff = df_agg[col_test] - df_agg[col_base]
        w_stat, p_val = wilcoxon(df_agg[col_test], df_agg[col_base], alternative='two-sided')
        ci = st.t.interval(0.95, len(diff)-1, loc=np.mean(diff), scale=st.sem(diff))
        
        print(f"\n[{name}]")
        print(f"  - Mean Difference: +{np.mean(diff):.2f} pp")
        print(f"  - 95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]")
        print(f"  - Wilcoxon p-value: {p_val:.4e} {'(Significant)' if p_val < 0.05 else '(Not Significant)'}")

    run_test('Dynamic_CART', 'Fixed_80_20', 'Dynamic vs Fixed')
    run_test('Dynamic_CART', 'Permuted_Trust', 'Dynamic vs Permuted')
    run_test('Dynamic_CART', 'Raw', 'Dynamic vs Raw')

if __name__ == "__main__":
    main()