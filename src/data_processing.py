import pandas as pd

# Load raw test
cols = ['unit', 'cycle', 'op1', 'op2', 'op3'] + [f'sensor_{i}' for i in range(1, 22)]
test = pd.read_csv("/Users/murad/nasa_proj/data/raw/test_FD001.txt", sep=r"\s+", header=None)
test.columns = cols

# OPTIONAL: minimal processing (for dashboard only)
# (no need full feature engineering just for plotting)

# Save
test.to_csv("/Users/murad/nasa_proj/data/processed/test_FD001.txt", index=False)

print("✅ Saved test_processed.csv")