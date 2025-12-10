import pandas as pd
from ucimlrepo import fetch_ucirepo
import os

SAVE_DIR = "data"
os.makedirs(SAVE_DIR, exist_ok=True)

def save_dataset(data, filename):
    X = data.data.features
    y = data.data.targets

    # Flatten multi-column targets
    if isinstance(y, pd.DataFrame) and y.shape[1] == 1:
        y = y.iloc[:, 0]

    df = X.copy()
    df["target"] = y

    out_path = os.path.join(SAVE_DIR, filename)
    df.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")


print("\nDownloading datasets from UCI...\n")

# ---------------------------------------
# HEART DISEASE (ID 45)
# ---------------------------------------
try:
    heart = fetch_ucirepo(id=45)
    save_dataset(heart, "heart.csv")
except Exception as e:
    print("Failed to download HEART dataset:", e)


# ---------------------------------------
# BANK MARKETING (ID 222)
# ---------------------------------------
try:
    bank = fetch_ucirepo(id=222)
    save_dataset(bank, "bank.csv")
except Exception as e:
    print("Failed to download BANK dataset:", e)


# ---------------------------------------
# BREAST CANCER (ID 17)
# ---------------------------------------
try:
    breast = fetch_ucirepo(id=17)
    save_dataset(breast, "breast.csv")
except Exception as e:
    print("Failed to download BREAST dataset:", e)


# ---------------------------------------
# PEN-BASED DIGITS (ID 81)
# ---------------------------------------
try:
    digits = fetch_ucirepo(id=81)
    save_dataset(digits, "digits.csv")
except Exception as e:
    print("Failed to download DIGITS dataset:", e)


# ---------------------------------------
# WINE QUALITY (ID 186)
# ---------------------------------------
try:
    wine = fetch_ucirepo(id=186)
    save_dataset(wine, "wine_quality.csv")
except Exception as e:
    print("Failed to download WINE QUALITY dataset:", e)


print("\nDone.\n")
