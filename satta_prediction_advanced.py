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
MARKETS = [
    "Time Bazar", "Milan Day", "Rajdhani Day",
    "Kalyan", "Milan Night", "Rajdhani Night", "Main Bazar"
]
DATA_FILE = "enhanced_satta_data.csv"
ACCURACY_FILE = "accuracy_log.csv"

# --- Telegram ---
def send_telegram_message(message):
    bot = telegram.Bot(token=TELEGRAM_TOKEN)
    bot.send_message(chat_id=CHAT_ID, text=message, parse_mode=telegram.ParseMode.HTML)

# --- Data Load ---
def load_data():
    df = pd.read_csv(DATA_FILE)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Market", "Open", "Close"])
    df["Open"] = pd.to_numeric(df["Open"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Open", "Close"])
    return df

# --- Feature Engineering ---
def engineer_features(df):
    df = df.sort_values("Date").copy()
    df["Prev_Open"] = df["Open"].shift(1)
    df["Prev_Close"] = df["Close"].shift(1)
    df["Weekday"] = df["Date"].dt.weekday
    df["Jodi"] = df["Open"].astype(int).astype(str) + df["Close"].astype(int).astype(str)
    df["Patti_Open"] = df["Open"].apply(lambda x: str(int(x)).zfill(3))
    df["Patti_Close"] = df["Close"].apply(lambda x: str(int(x)).zfill(3))
    df = df.dropna()
    return df

# --- Digit conversion ---
def patti_to_digit(patti):
    return sum(int(d) for d in str(int(patti)).zfill(3)) % 10

def generate_jodis(open_digits, close_digits):
    return list(set([f"{o}{c}"[-2:] for o in open_digits for c in close_digits]))[:10]

def generate_pattis(open_vals, close_vals):
    pattis = set()
    for val in open_vals + close_vals:
        try:
            base = int(val)
            pattis.update([str(base + i).zfill(3) for i in range(4)])
        except:
            continue
    return list(pattis)[:4]

# --- Model Training & Prediction ---
def train_model(X, y):
    model = RandomForestClassifier()
    grid = GridSearchCV(model, {
        'n_estimators': [50, 100],
        'max_depth': [None, 5, 10]
    }, cv=3)
    grid.fit(X, y)
    return grid.best_estimator_

def train_and_predict(df, market):
    df_market = df[df["Market"] == market].copy()
    df_market = engineer_features(df_market)
    if len(df_market) < 30:
        return None, None, None, None

    tomorrow = df_market["Date"].max() + timedelta(days=1)
    weekday = tomorrow.weekday()
    last_row = df_market.iloc[-1]

    X = df_market[["Prev_Open", "Prev_Close", "Weekday"]]
    y_open = df_market["Open"].astype(int)
    y_close = df_market["Close"].astype(int)

    X_train, _, y_train_open, _ = train_test_split(X, y_open, test_size=0.2, random_state=42)
    X_train2, _, y_train_close, _ = train_test_split(X, y_close, test_size=0.2, random_state=42)

    model_open = train_model(X_train, y_train_open)
    model_close = train_model(X_train2, y_train_close)

    X_pred = pd.DataFrame([{
        "Prev_Open": last_row["Open"],
        "Prev_Close": last_row["Close"],
        "Weekday": weekday
    }])

    open_probs = model_open.predict_proba(X_pred)[0]
    close_probs = model_close.predict_proba(X_pred)[0]

    open_classes = model_open.classes_
    close_classes = model_close.classes_

    open_vals = [open_classes[i] for i in np.argsort(open_probs)[-2:][::-1]]
    close_vals = [close_classes[i] for i in np.argsort(close_probs)[-2:][::-1]]

    return open_vals, close_vals, model_open, model_close

# --- Accuracy Logging ---
def log_accuracy(market, actual_open, actual_close, pred_open, pred_close):
    matched_open = int(actual_open) in pred_open
    matched_close = int(actual_close) in pred_close
    matched_jodi = str(actual_open)[-1] + str(actual_close)[-1] in generate_jodis([str(patti_to_digit(o)) for o in pred_open], [str(patti_to_digit(c)) for c in pred_close])
    matched_patti = str(actual_open).zfill(3) in generate_pattis(pred_open, pred_close) or str(actual_close).zfill(3) in generate_pattis(pred_open, pred_close)

    row = {
        "Date": datetime.now().strftime("%Y-%m-%d"),
        "Market": market,
        "Open_Match": int(matched_open),
        "Close_Match": int(matched_close),
        "Jodi_Match": int(matched_jodi),
        "Patti_Match": int(matched_patti)
    }

    if os.path.exists(ACCURACY_FILE):
        df_log = pd.read_csv(ACCURACY_FILE)
        df_log = df_log[df_log["Market"] != market]
        df_log = pd.concat([df_log, pd.DataFrame([row])])
    else:
        df_log = pd.DataFrame([row])

    df_log.to_csv(ACCURACY_FILE, index=False)

# --- Main Pipeline ---
def main():
    df = load_data()
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%d/%m/%Y")
    full_message = f"<b>Tomorrow's Predictions ({tomorrow}):</b>\n"

    for market in MARKETS:
        open_vals, close_vals, model_open, model_close = train_and_predict(df, market)
        if not open_vals or not close_vals:
            full_message += f"\n<b>{market}</b>\n<i>Prediction Failed or Not Enough Data</i>\n"
            continue

        open_digits = [str(patti_to_digit(val)) for val in open_vals]
        close_digits = [str(patti_to_digit(val)) for val in close_vals]
        jodis = generate_jodis(open_digits, close_digits)
        pattis = generate_pattis(open_vals, close_vals)

        full_message += (
            f"\n<b>{market}</b>\n"
            f"<code>{tomorrow}</code>\n"
            f"<b>Open:</b> {', '.join(open_digits)}\n"
            f"<b>Close:</b> {', '.join(close_digits)}\n"
            f"<b>Jodi:</b> {', '.join(jodis)}\n"
            f"<b>Patti:</b> {', '.join(pattis)}\n"
        )

    send_telegram_message(full_message)

if __name__ == "__main__":
    main()
