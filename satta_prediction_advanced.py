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
DATA_FILE = "enhanced_satta_data.csv"

# --- Telegram ---
def send_telegram_message(message):
    bot = telegram.Bot(token=TELEGRAM_TOKEN)
    bot.send_message(chat_id=CHAT_ID, text=message, parse_mode=telegram.ParseMode.HTML)

# --- Load and clean data ---
def load_data():
    df = pd.read_csv(DATA_FILE)
    df = df[["Date", "Market", "Open", "Close", "Open Patti", "Jodi", "Close Patti"]]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna()
    df["Open"] = pd.to_numeric(df["Open"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    return df.dropna()

# --- Feature Engineering ---
def prepare_features(df_market):
    df = df_market.sort_values("Date").copy()
    df["Prev_Open"] = df["Open"].shift(1)
    df["Prev_Close"] = df["Close"].shift(1)
    df["Weekday"] = df["Date"].dt.weekday
    df = df.dropna()
    return df

def patti_to_digit(patti):
    return str(sum(int(d) for d in str(patti).zfill(3)) % 10)

def generate_jodis(open_digits, close_digits):
    return list(set([f"{o}{c}"[-2:] for o in open_digits for c in close_digits]))[:10]

def generate_pattis(patti_list):
    pattis = set()
    for val in patti_list:
        try:
            base = int(val)
            pattis.update([str(base + i).zfill(3) for i in range(4)])
        except:
            continue
    return list(pattis)[:4]

def train_model(X, y):
    if len(X) < 5:
        return None
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X, y)
    return model

def train_and_predict(df, market):
    df_market = df[df["Market"] == market]
    if len(df_market) < 6:
        return None

    df_market = prepare_features(df_market)
    if len(df_market) < 5:
        return None

    last_row = df_market.iloc[-1]
    tomorrow_weekday = (df_market["Date"].max() + timedelta(days=1)).weekday()

    X = df_market[["Prev_Open", "Prev_Close", "Weekday"]]
    y_open_patti = df_market["Open Patti"].astype(str).str.zfill(3)
    y_close_patti = df_market["Close Patti"].astype(str).str.zfill(3)

    model_open = train_model(X, y_open_patti)
    model_close = train_model(X, y_close_patti)

    if model_open is None or model_close is None:
        return None

    X_pred = pd.DataFrame([{
        "Prev_Open": last_row["Open"],
        "Prev_Close": last_row["Close"],
        "Weekday": tomorrow_weekday
    }])

    open_probs = model_open.predict_proba(X_pred)[0]
    close_probs = model_close.predict_proba(X_pred)[0]

    open_classes = model_open.classes_
    close_classes = model_close.classes_

    top_open = [open_classes[i] for i in np.argsort(open_probs)[-2:][::-1]]
    top_close = [close_classes[i] for i in np.argsort(close_probs)[-2:][::-1]]

    return top_open, top_close

# --- Main ---
def main():
    df = load_data()
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%d/%m/%Y")
    full_msg = f"<b>Tomorrow's Predictions ({tomorrow}):</b>\n"

    for market in MARKETS:
        result = train_and_predict(df, market)
        if result is None:
            full_msg += f"\n<b>{market}</b>\n<i>Not enough data for prediction</i>\n"
            continue

        top_open, top_close = result

        open_digits = [patti_to_digit(p) for p in top_open]
        close_digits = [patti_to_digit(p) for p in top_close]

        jodis = generate_jodis(open_digits, close_digits)
        pattis = generate_pattis(top_open + top_close)

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
