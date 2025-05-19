import pandas as pd
import numpy as np
import telegram
import os
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

# --- Config ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "7121966371:AAEKHVrsqLRswXg64-6Nf3nid-Mbmlmmw5M")
CHAT_ID = os.getenv("CHAT_ID", "7621883960")
MARKETS = ["Time Bazar", "Milan Day", "Rajdhani Day", "Kalyan", "Milan Night", "Rajdhani Night", "Main Bazar"]
DATA_FILE = "enhanced_satta_data.csv"
ACCURACY_FILE = "accuracy_log.csv"

# --- Telegram ---
def send_telegram_message(message):
    bot = telegram.Bot(token=TELEGRAM_TOKEN)
    bot.send_message(chat_id=CHAT_ID, text=message, parse_mode=telegram.ParseMode.HTML)

# --- Load Data ---
def load_data():
    df = pd.read_csv(DATA_FILE)
    df["Market"] = df["Market"].astype(str).str.strip()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["Date", "Market", "Open", "Close"])
    df["Open"] = pd.to_numeric(df["Open"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna()
    return df

# --- Feature Engineering ---
def engineer_features(df):
    df = df.sort_values("Date").copy()
    df["Prev_Open"] = df["Open"].shift(1)
    df["Prev_Close"] = df["Close"].shift(1)
    df["Weekday"] = df["Date"].dt.weekday
    df["Is_Weekend"] = df["Weekday"] >= 5
    df["Holiday_Flag"] = df["is_holiday"].fillna(False).astype(int)
    df["Prev_Jodi_Dist"] = df["prev_jodi_distance"].fillna(0)
    df["Open_Sum"] = df["open_sum"].fillna(0)
    df["Close_Sum"] = df["close_sum"].fillna(0)
    return df.dropna()

# --- Utility ---
def patti_to_digit(patti):
    return str(sum(int(d) for d in str(int(patti)).zfill(3)) % 10)

def generate_jodis(open_digits, close_digits):
    return list(set([f"{o}{c}"[-2:] for o in open_digits for c in close_digits]))[:10]

def generate_pattis(vals):
    pattis = set()
    for val in vals:
        try:
            base = int(val)
            pattis.update([str(base + i).zfill(3) for i in range(4)])
        except:
            continue
    return list(pattis)[:4]

# --- Model Training ---
def train_model(X, y):
    clf = RandomForestClassifier()
    grid = GridSearchCV(clf, {
        "n_estimators": [100],
        "max_depth": [10, None]
    }, cv=3)
    grid.fit(X, y)
    return grid.best_estimator_

# --- Predict ---
def train_and_predict(df, market):
    df_market = df[df["Market"] == market].copy()
    if len(df_market) < 20: return None, None, None, None

    df_market = engineer_features(df_market)
    last_row = df_market.iloc[-1]
    tomorrow = df_market["Date"].max() + timedelta(days=1)
    weekday = tomorrow.weekday()

    features = ["Prev_Open", "Prev_Close", "Weekday", "Is_Weekend", "Holiday_Flag", "Prev_Jodi_Dist", "Open_Sum", "Close_Sum"]
    X = df_market[features]
    y_open = df_market["Open"].astype(int)
    y_close = df_market["Close"].astype(int)

    X_train, _, y_train_open, _ = train_test_split(X, y_open, test_size=0.2, random_state=42)
    X_train2, _, y_train_close, _ = train_test_split(X, y_close, test_size=0.2, random_state=42)

    model_open = train_model(X_train, y_train_open)
    model_close = train_model(X_train2, y_train_close)

    X_pred = pd.DataFrame([{
        "Prev_Open": last_row["Open"],
        "Prev_Close": last_row["Close"],
        "Weekday": weekday,
        "Is_Weekend": int(weekday >= 5),
        "Holiday_Flag": 0,
        "Prev_Jodi_Dist": last_row["prev_jodi_distance"],
        "Open_Sum": last_row["open_sum"],
        "Close_Sum": last_row["close_sum"]
    }])

    open_probs = model_open.predict_proba(X_pred)[0]
    close_probs = model_close.predict_proba(X_pred)[0]

    open_classes = model_open.classes_
    close_classes = model_close.classes_

    open_vals = [open_classes[i] for i in np.argsort(open_probs)[-2:][::-1]]
    close_vals = [close_classes[i] for i in np.argsort(close_probs)[-2:][::-1]]

    return open_vals, close_vals, model_open, model_close

# --- Accuracy Logging ---
def update_accuracy_log(market, date, predicted, actual):
    acc_df = pd.DataFrame([{
        "Market": market,
        "Date": date,
        "Open_Pred": predicted["open"],
        "Close_Pred": predicted["close"],
        "Open_Actual": actual["open"],
        "Close_Actual": actual["close"],
        "Jodi_Pred": predicted["jodi"],
        "Jodi_Actual": actual["jodi"],
        "Patti_Pred": ",".join(predicted["patti"]),
        "Patti_Actual": actual["patti"]
    }])
    if os.path.exists(ACCURACY_FILE):
        old = pd.read_csv(ACCURACY_FILE)
        acc_df = pd.concat([old, acc_df], ignore_index=True)
    acc_df.to_csv(ACCURACY_FILE, index=False)

# --- Main ---
def main():
    df = load_data()
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%d/%m/%Y")
    full_msg = f"<b>Tomorrow's Predictions ({tomorrow}):</b>\n"

    for market in MARKETS:
        open_vals, close_vals, _, _ = train_and_predict(df, market)
        if not open_vals or not close_vals:
            full_msg += f"\n<b>{market}</b>\n<i>Prediction Failed or Insufficient Data</i>\n"
            continue

        open_digits = [patti_to_digit(val) for val in open_vals]
        close_digits = [patti_to_digit(val) for val in close_vals]
        jodis = generate_jodis(open_digits, close_digits)
        pattis = generate_pattis(open_vals + close_vals)

        full_msg += (
            f"\n<b>{market}</b>\n"
            f"<code>{tomorrow}</code>\n"
            f"<b>Open:</b> {', '.join(open_digits)}\n"
            f"<b>Close:</b> {', '.join(close_digits)}\n"
            f"<b>Jodi:</b> {', '.join(jodis)}\n"
            f"<b>Patti:</b> {', '.join(pattis)}\n"
        )

    send_telegram_message(full_msg)

if __name__ == "__main__":
    main()
