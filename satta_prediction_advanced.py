import pandas as pd
import numpy as np
import telegram
from datetime import datetime, timedelta
import warnings
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

# Telegram credentials
TELEGRAM_TOKEN = "7121966371:AAEKHVrsqLRswXg64-6Nf3nid-Mbmlmmw5M"
CHAT_ID = "7621883960"
bot = telegram.Bot(token=TELEGRAM_TOKEN)

# File paths
CSV_PATH = "enhanced_satta_data.csv"
ACCURACY_LOG = "accuracy_log.csv"

# Market list
MARKETS = ["Time Bazar", "Milan Day", "Rajdhani Day", "Kalyan", "Milan Night", "Rajdhani Night", "Main Bazar"]

def clean_data(df):
    df = df.dropna(subset=["Market", "Date", "Open", "Close"])
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date")
    df["Jodi"] = df["Open"].astype(str) + df["Close"].astype(str)
    df["Patti"] = df["Open"].astype(str).str.zfill(1) + df["Close"].astype(str).str.zfill(2)
    df["Weekday"] = df["Date"].dt.dayofweek
    return df

def get_features_and_labels(df, col):
    df["Prev_" + col] = df[col].shift(1)
    df = df.dropna()
    X = df[["Prev_" + col, "Weekday"]]
    y = df[col]
    return X, y

def train_and_predict(df, market, col):
    df = df[df["Market"] == market]
    if len(df) < 20:
        return None, f"{market}: Not enough data for training"
    
    X, y = get_features_and_labels(df, col)
    if X.empty or y.empty or len(X) < 10:
        return None, f"{market}: Not enough valid rows after feature prep"

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        tomorrow_weekday = (df["Date"].max() + timedelta(days=1)).weekday()
        last_val = df[col].iloc[-1]
        pred = model.predict([[last_val, tomorrow_weekday]])[0]
        return pred, acc
    except Exception as e:
        return None, f"{market}: Prediction failed ({str(e)})"

def send_telegram_message(message):
    bot.send_message(chat_id=CHAT_ID, text=message, parse_mode=telegram.ParseMode.HTML)

def log_accuracy(market, col, acc):
    row = {
        "Date": datetime.now().strftime("%Y-%m-%d"),
        "Market": market,
        "Field": col,
        "Accuracy": acc
    }
    if not os.path.exists(ACCURACY_LOG):
        pd.DataFrame([row]).to_csv(ACCURACY_LOG, index=False)
    else:
        pd.concat([pd.read_csv(ACCURACY_LOG), pd.DataFrame([row])], ignore_index=True).to_csv(ACCURACY_LOG, index=False)

def main():
    if not os.path.exists(CSV_PATH):
        print("CSV not found.")
        return

    df = pd.read_csv(CSV_PATH)
    df = clean_data(df)

    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    full_message = f"<b>Tomorrow's Predictions ({tomorrow}):</b>\n"

    for market in MARKETS:
        try:
            open_pred, open_acc = train_and_predict(df, market, "Open")
            close_pred, close_acc = train_and_predict(df, market, "Close")
            jodi_pred = f"{open_pred}{close_pred}" if open_pred and close_pred else "N/A"
            if isinstance(open_pred, str) and open_pred.startswith(market):
                full_message += f"\n<b>{open_pred}</b>"
                continue

            full_message += f"\n<b>{market}</b>:\nOpen: <b>{open_pred}</b>, Close: <b>{close_pred}</b>, Jodi: <b>{jodi_pred}</b>"
            log_accuracy(market, "Open", open_acc)
            log_accuracy(market, "Close", close_acc)

        except Exception as e:
            full_message += f"\n<b>{market}</b>: Prediction error - {str(e)}"

    send_telegram_message(full_message)

if __name__ == "__main__":
    main()
