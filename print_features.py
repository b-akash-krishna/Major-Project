import sys
import os
import joblib

sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
from config import MAIN_MODEL_PKL, MAIN_MODEL_PKL_LEGACY

if os.path.exists(MAIN_MODEL_PKL):
    pkl_path = MAIN_MODEL_PKL
else:
    pkl_path = MAIN_MODEL_PKL_LEGACY
data = joblib.load(pkl_path)
features = data.get('features', [])
ct5 = [x for x in features if x.startswith('ct5_')]
tab = [x for x in features if not x.startswith('ct5_')]

with open('features_output.txt', 'w') as f:
    f.write(f"Total features: {len(features)}\n")
    f.write(f"Embedding features (ct5_*): {len(ct5)}\n")
    f.write(f"Tabular features: {len(tab)}\n")
    f.write(f"\nTabular Feature List:\n{', '.join(tab)}\n")
