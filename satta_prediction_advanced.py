import pandas as pd
import numpy as np
import telegram
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# Telegram credentials
TELEGRAM_TOKEN = "7121966371:AAEKHVrsqLRswXg64-6Nf3nid-Mbmlmmw5M"
CHAT_ID = "7621883960"

MARKETS = [
    "Time Bazar", "Milan Day", "Rajdhani Day", 
    "Kalyan", "Milan Night", "Rajdhani Night", "Main Bazar"
]

def send_telegram_message(message):
    bot = telegram.Bot(token=TELEGRAM_TOKEN)
    bot.send_message(chat_id=CHAT_ID, text=message, parse_mode=telegram.ParseMode.HTML)

def load_data():
    df = pd.read_csv("enhanced_satta_data.csv")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "Market", "Open", "Close"])
    df["Market"] = df["Market"].astype(str).str.strip()
    df["Open"] = pd.to_numeric(df["Open"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    return df

def engineer_features(df_market):
    df_market = df_market.sort_values("Date")
    df_market["Prev_Open"] = df_market["Open"].shift(1)
    df_market["Prev_Close"] = df_market["Close"].shift(1)
    df_market["Weekday"] = df_market["Date"].dt.weekday
    df_market = df_market.dropna()
    return df_market

def generate_jodis(open_vals, close_vals):
    return list(set([f"{o}{c}"[-2:] for o in open_vals for c in close_vals]))[:10]

def generate_pattis(open_vals, close_vals):
    pattis = set()
    for val in open_vals + close_vals:
        base = int(val)
        pattis.update([str(base + i).zfill(3) for i in range(4)])
    return list(pattis)[:4]

def train_and_predict(df, market):
    df_market = df[df["Market"] == market].copy()
    df_market = engineer_features(df_market)

    if len(df_market) < 20:
        return None, None

    tomorrow = df_market["Date"].max() + timedelta(days=1)
    weekday = tomorrow.weekday()

    features = ["Prev_Open", "Prev_Close", "Weekday"]
    X = df_market[features]
    y_open = df_market["Open"].astype(int)
    y_close = df_market["Close"].astype(int)

    try:
        X_train, _, y_train_open, _ = train_test_split(X, y_open, test_size=0.2, random_state=42)
        model_open = RandomForestClassifier(n_estimators=100, random_state=42)
        model_open.fit(X_train, y_train_open)

        _, _, y_train_close, _ = train_test_split(X, y_close, test_size=0.2, random_state=42)
        model_close = RandomForestClassifier(n_estimators=100, random_state=42)
        model_close.fit(X_train, y_train_close)

        last_row = df_market.iloc[-1]
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
    except Exception as e:
        print(f"{market} prediction error: {e}")
        return None, None

def main():
    df = load_data()
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%d/%m/%Y")
    full_message = f"<b>Tomorrow's Predictions ({tomorrow}):</b>\n"

    for market in MARKETS:
        open_vals, close_vals = train_and_predict(df, market)
        if not open_vals or not close_vals:
            full_message += f"\n<b>{market}</b>\n<i>Prediction Failed or Not Enough Data</i>\n"
            continue

        jodis = generate_jodis(open_vals, close_vals)
        pattis = generate_pattis(open_vals, close_vals)

        full_message += (
            f"\n<b>{market}</b>\n"
            f"<code>{tomorrow}</code>\n"
            f"<b>Open:</b> {', '.join(map(str, open_vals))}\n"
            f"<b>Close:</b> {', '.join(map(str, close_vals))}\n"
            f"<b>Jodi:</b> {', '.join(jodis)}\n"
            f"<b>Patti:</b> {', '.join(pattis)}\n"
        )

    send_telegram_message(full_message)

if __name__ == "__main__":
    main()
