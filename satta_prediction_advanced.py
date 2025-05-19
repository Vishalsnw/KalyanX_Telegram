import pandas as pd
import numpy as np
import telegram
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

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
    df = df.dropna(subset=["Open", "Close"])
    return df


def engineer_features(df_market):
    df_market = df_market.sort_values("Date").copy()
    df_market["Prev_Open"] = df_market["Open"].shift(1)
    df_market["Prev_Close"] = df_market["Close"].shift(1)
    df_market["Weekday"] = df_market["Date"].dt.weekday
    df_market = df_market.iloc[1:]
    return df_market


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


def calculate_near_miss(pred, actual):
    return abs(int(pred) - int(actual)) <= 5


def is_flip(pred, actual):
    return str(pred)[::-1] == str(actual)


def feedback_and_weights(y_true, y_pred):
    score_map = []
    for true, pred in zip(y_true, y_pred):
        if str(true) == str(pred):
            score_map.append(2)
        elif is_flip(pred, true):
            score_map.append(1.5)
        elif calculate_near_miss(pred, true):
            score_map.append(1)
        else:
            score_map.append(0.5)
    return score_map


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

        open_scores = [(open_classes[i], open_probs[i]) for i in range(len(open_classes))]
        close_scores = [(close_classes[i], close_probs[i]) for i in range(len(close_classes))]

        open_scores = sorted(open_scores, key=lambda x: x[1], reverse=True)
        close_scores = sorted(close_scores, key=lambda x: x[1], reverse=True)

        open_vals = [int(o[0]) for o in open_scores[:2]]
        close_vals = [int(c[0]) for c in close_scores[:2]]

        return open_vals, close_vals
    except Exception as e:
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
        
