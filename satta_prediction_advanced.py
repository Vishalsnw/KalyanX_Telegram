import pandas as pd
import numpy as np
import telegram
import os
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")

# --- Config ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "7121966371:AAEKHVrsqLRswXg64-6Nf3nid-Mbmlmmw5M")
CHAT_ID = os.getenv("CHAT_ID", "7621883960")
MARKETS = ["Time Bazar", "Milan Day", "Rajdhani Day", "Kalyan", "Milan Night", "Rajdhani Night", "Main Bazar"]
DATA_FILE = "satta_data.csv"
ACCURACY_LOG = "accuracy_log.csv"

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
def engineer_features(df_market):
    df = df_market.sort_values("Date").copy()
    df["Prev_Open"] = df["Open"].shift(1)
    df["Prev_Close"] = df["Close"].shift(1)
    df["Weekday"] = df["Date"].dt.weekday
    df = df.dropna(subset=["Prev_Open", "Prev_Close"])
    return df

# --- Utilities ---
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

def flip_number(jodi):
    return jodi[::-1] if len(jodi) == 2 else jodi

def is_near_miss(actual, predicted_jodis):
    actual = int(actual)
    for pred in predicted_jodis:
        try:
            if abs(int(pred) - actual) <= 5:
                return True
        except:
            continue
    return False

# --- Model Trainer ---
def train_model(X, y):
    if len(X) < 5:
        return None
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    return model

# --- Predict One Market ---
def train_and_predict(df, market):
    df_market = df[df["Market"] == market].copy()
    if df_market.shape[0] < 6:
        return None, None

    df_market = engineer_features(df_market)
    if len(df_market) < 5:
        return None, None

    last_row = df_market.iloc[-1]
    tomorrow = df_market["Date"].max() + timedelta(days=1)
    weekday = tomorrow.weekday()

    X = df_market[["Prev_Open", "Prev_Close", "Weekday"]]
    y_open = df_market["Open"].astype(int)
    y_close = df_market["Close"].astype(int)

    if len(X) != len(y_open) or len(X) != len(y_close):
        return None, None

    model_open = train_model(X, y_open)
    model_close = train_model(X, y_close)

    if model_open is None or model_close is None:
        return None, None

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

    return open_vals, close_vals

# --- Accuracy Logger ---
def check_results_and_log_accuracy(df, predictions):
    today = datetime.now().date()
    results_today = df[df["Date"].dt.date == today]
    logs = []

    for market, (open_vals, close_vals) in predictions.items():
        df_market = results_today[results_today["Market"] == market]
        if df_market.empty:
            continue

        row = df_market.iloc[0]
        actual_open = int(row["Open"])
        actual_close = int(row["Close"])
        actual_jodi = f"{actual_open}{actual_close}"[-2:]

        open_digits = [str(patti_to_digit(val)) for val in open_vals]
        close_digits = [str(patti_to_digit(val)) for val in close_vals]
        predicted_jodis = generate_jodis(open_digits, close_digits)

        log = {
            "Date": today.strftime("%d/%m/%Y"),
            "Market": market,
            "Actual_Jodi": actual_jodi,
            "Predicted_Jodis": ', '.join(predicted_jodis),
            "Exact_Match": actual_jodi in predicted_jodis,
            "Near_Miss": is_near_miss(actual_jodi, predicted_jodis),
            "Flip_Match": flip_number(actual_jodi) in predicted_jodis
        }
        logs.append(log)

    if logs:
        df_logs = pd.DataFrame(logs)
        if os.path.exists(ACCURACY_LOG):
            df_logs.to_csv(ACCURACY_LOG, mode="a", header=False, index=False)
        else:
            df_logs.to_csv(ACCURACY_LOG, index=False)

# --- Main ---
def main():
    df = load_data()
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%d/%m/%Y")
    full_msg = f"<b>Tomorrow's Predictions ({tomorrow}):</b>\n"
    predictions = {}

    for market in MARKETS:
        open_vals, close_vals = train_and_predict(df, market)
        if not open_vals or not close_vals:
            full_msg += f"\n<b>{market}</b>\n<i>Not enough data</i>\n"
            continue

        predictions[market] = (open_vals, close_vals)

        open_digits = [str(patti_to_digit(val)) for val in open_vals]
        close_digits = [str(patti_to_digit(val)) for val in close_vals]
        jodis = generate_jodis(open_digits, close_digits)
        pattis = generate_pattis(open_vals, close_vals)

        full_msg += (
            f"\n<b>{market}</b>\n"
            f"<b>Open:</b> {', '.join(open_digits)}\n"
            f"<b>Close:</b> {', '.join(close_digits)}\n"
            f"<b>Jodi:</b> {', '.join(jodis)}\n"
            f"<b>Patti:</b> {', '.join(pattis)}\n"
        )

    send_telegram_message(full_msg)

    # Check accuracy if results available
    check_results_and_log_accuracy(df, predictions)

if __name__ == "__main__":
    main()
