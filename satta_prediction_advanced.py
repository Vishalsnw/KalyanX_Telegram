import pandas as pd
import numpy as np
import telegram
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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
    df = df.dropna(subset=["Date", "Market"])
    df["Market"] = df["Market"].astype(str).str.strip()
    df["Open"] = pd.to_numeric(df["Open"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    return df

def engineer_features(df_market):
    df_market = df_market.sort_values("Date")
    df_market["Prev_Open"] = df_market["Open"].shift(1)
    df_market["Prev_Close"] = df_market["Close"].shift(1)
    df_market["Weekday"] = df_market["Date"].dt.weekday
    df_market = df_market.dropna(subset=["Prev_Open", "Prev_Close", "Open", "Close"])
    return df_market

def train_and_predict(df, market):
    df_market = df[df["Market"] == market].copy()
    df_market = engineer_features(df_market)

    if len(df_market) < 10:
        print(f"{market}: Not enough valid data after feature engineering. Rows: {len(df_market)}")
        return None, None

    # Predict next day
    tomorrow = df_market["Date"].max() + timedelta(days=1)
    weekday = tomorrow.weekday()

    # Prepare data
    features = ["Prev_Open", "Prev_Close", "Weekday"]
    X = df_market[features]
    y_open = df_market["Open"].astype(int)
    y_close = df_market["Close"].astype(int)

    if X.isnull().any().any() or y_open.isnull().any() or y_close.isnull().any():
        print(f"{market}: Found NaNs in training data.")
        return None, None

    try:
        X_train, X_test, y_train_open, y_test_open = train_test_split(X, y_open, test_size=0.2, random_state=42)
        model_open = RandomForestClassifier(n_estimators=100, random_state=42)
        model_open.fit(X_train, y_train_open)

        y_train_close, y_test_close = train_test_split(y_close, test_size=0.2, random_state=42)
        model_close = RandomForestClassifier(n_estimators=100, random_state=42)
        model_close.fit(X_train, y_train_close)

        # Predict for tomorrow
        last_row = df_market.iloc[-1]
        X_pred = pd.DataFrame([{
            "Prev_Open": last_row["Open"],
            "Prev_Close": last_row["Close"],
            "Weekday": weekday
        }])

        open_pred = int(model_open.predict(X_pred)[0])
        close_pred = int(model_close.predict(X_pred)[0])
        return open_pred, close_pred

    except Exception as e:
        print(f"{market} prediction error: {e}")
        return None, None

def main():
    df = load_data()
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    full_message = f"<b>Tomorrow's Predictions ({tomorrow}):</b>\n"

    for market in MARKETS:
        try:
            open_pred, close_pred = train_and_predict(df, market)
            if open_pred is None or close_pred is None:
                full_message += f"\n<b>{market}:</b> Not enough data to predict"
            else:
                jodi = f"{open_pred}{close_pred}"[-2:]
                full_message += f"\n<b>{market}</b>:\nOpen: {open_pred}, Close: {close_pred}, Jodi: {jodi}"
        except Exception as e:
            full_message += f"\n<b>{market}:</b> Prediction failed ({str(e)})"

    print(full_message)
    send_telegram_message(full_message)

if __name__ == "__main__":
    main()
