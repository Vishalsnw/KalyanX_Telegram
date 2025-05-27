import pandas as pd
import numpy as np
import telegram
import os
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from dotenv import load_dotenv
import requests
import warnings

warnings.filterwarnings("ignore")
load_dotenv()

# --- Config ---
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
MARKETS = ["Time Bazar", "Milan Day", "Rajdhani Day", "Kalyan", "Milan Night", "Rajdhani Night", "Main Bazar"]
DATA_FILE = "satta_data.csv"
PRED_FILE = "today_ml_prediction.csv"
COPILOT_API_KEY = "a531e727f3msh281ef1f076f7139p198608jsn82cfb1c7b6d0"

# --- Telegram ---
def send_telegram_message(message):
    bot = telegram.Bot(token=TELEGRAM_TOKEN)
    bot.send_message(chat_id=CHAT_ID, text=message, parse_mode=telegram.ParseMode.HTML)

# --- Load Data ---
def load_data():
    df = pd.read_csv(DATA_FILE)
    df["Market"] = df["Market"].astype(str).str.strip()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["Date", "Market", "Open", "Close", "Jodi"])
    df["Open"] = pd.to_numeric(df["Open"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["Jodi"] = df["Jodi"].astype(str).str.zfill(2).str[-2:]
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

def generate_pattis(open_vals, close_vals):
    pattis = set()
    for val in open_vals + close_vals:
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

# --- Copilot API ---
def enhance_with_copilot(open_vals, close_vals, jodi_vals):
    message = {
        "message": f"Given Open: {open_vals}, Close: {close_vals}, Jodi: {jodi_vals}. Suggest improved predictions to increase satta model accuracy.",
        "conversation_id": None,
        "mode": "CHAT",
        "markdown": False
    }
    headers = {
        "Content-Type": "application/json",
        "x-rapidapi-host": "copilot5.p.rapidapi.com",
        "x-rapidapi-key": COPILOT_API_KEY
    }
    try:
        res = requests.post("https://copilot5.p.rapidapi.com/copilot", json=message, headers=headers, timeout=10)
        if res.status_code == 200:
            data = res.json()
            suggestion = data.get("response", {}).get("text", "")
            return suggestion
    except Exception:
        pass
    return ""

def train_and_predict(df, market):
    df_market = df[df["Market"] == market].copy()
    if df_market.shape[0] < 6:
        return None, None, None, ""

    df_market = engineer_features(df_market)
    if len(df_market) < 5:
        return None, None, None, ""

    last_row = df_market.iloc[-1]
    tomorrow = df_market["Date"].max() + timedelta(days=1)
    weekday = tomorrow.weekday()

    X = df_market[["Prev_Open", "Prev_Close", "Weekday"]]
    y_open = df_market["Open"].astype(int)
    y_close = df_market["Close"].astype(int)

    model_open = train_model(X, y_open)
    model_close = train_model(X, y_close)

    if model_open is None or model_close is None:
        return None, None, None, ""

    X_pred = pd.DataFrame([{ "Prev_Open": last_row["Open"], "Prev_Close": last_row["Close"], "Weekday": weekday }])
    open_probs = model_open.predict_proba(X_pred)[0]
    close_probs = model_close.predict_proba(X_pred)[0]
    open_classes = model_open.classes_
    close_classes = model_close.classes_

    open_vals = [open_classes[i] for i in np.argsort(open_probs)[-2:][::-1]]
    close_vals = [close_classes[i] for i in np.argsort(close_probs)[-2:][::-1]]

    df_jodi = df_market[["Prev_Open", "Prev_Close", "Weekday", "Jodi"]].dropna()
    X_jodi = df_jodi[["Prev_Open", "Prev_Close", "Weekday"]]
    y_jodi = df_jodi["Jodi"]
    model_jodi = train_model(X_jodi, y_jodi)
    jodi_vals = []
    if model_jodi:
        jodi_probs = model_jodi.predict_proba(X_pred)[0]
        jodi_classes = model_jodi.classes_
        top_jodis = [jodi_classes[i] for i in np.argsort(jodi_probs)[-10:][::-1]]
        jodi_vals = top_jodis

    suggestion = enhance_with_copilot(open_vals, close_vals, jodi_vals)

    return open_vals, close_vals, jodi_vals, suggestion

# --- Main ---
def main():
    df = load_data()
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%d/%m/%Y")
    full_msg = f"<b>Tomorrow's Predictions ({tomorrow}):</b>\n"

    try:
        df_existing = pd.read_csv(PRED_FILE)
    except FileNotFoundError:
        df_existing = pd.DataFrame()

    new_rows = []

    for market in MARKETS:
        open_vals, close_vals, jodis, ai_suggestion = train_and_predict(df, market)
        if not open_vals or not close_vals or not jodis:
            full_msg += f"\n<b>{market}</b>\n<i>Not enough data</i>\n"
            continue

        open_digits = [str(patti_to_digit(val)) for val in open_vals]
        close_digits = [str(patti_to_digit(val)) for val in close_vals]
        pattis = generate_pattis(open_vals, close_vals)

        full_msg += (
            f"\n<b>{market}</b>\n"
            f"<b>Open:</b> {', '.join(open_digits)}\n"
            f"<b>Close:</b> {', '.join(close_digits)}\n"
            f"<b>Patti:</b> {', '.join(pattis)}\n"
            f"<b>Jodi:</b> {', '.join(jodis)}\n"
            f"<i>AI Suggestion:</i> {ai_suggestion[:100]}...\n"
        )

        new_rows.append({
            "Market": market,
            "Date": tomorrow,
            "Open": ", ".join(open_digits),
            "Close": ", ".join(close_digits),
            "Pattis": ", ".join(pattis),
            "Jodis": ", ".join(jodis)
        })

    for row in new_rows:
        df_existing = df_existing[~(
            (df_existing['Market'] == row['Market']) & 
            (df_existing['Date'] == row['Date'])
        )]

    df_combined = pd.concat([df_existing, pd.DataFrame(new_rows)], ignore_index=True)
    df_combined.to_csv(PRED_FILE, index=False)

    send_telegram_message(full_msg)

if __name__ == "__main__":
    main()
  
