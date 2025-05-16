import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from telegram import Bot
from telegram.constants import ParseMode
import os
import numpy as np

# Telegram settings
TOKEN = os.getenv("TOKEN", "your_default_token")
CHAT_ID = os.getenv("CHAT_ID", "your_default_chat_id")
bot = Bot(token=TOKEN)

# Source URLs
MARKET_URLS = {
    "Time Bazar": "https://dpbossattamatka.com/panel-chart-record/time-bazar.php",
    "Milan Day": "https://dpbossattamatka.com/panel-chart-record/milan-day.php",
    "Rajdhani Day": "https://dpbossattamatka.com/panel-chart-record/rajdhani-day.php",
    "Kalyan": "https://dpbossattamatka.com/panel-chart-record/kalyan.php",
    "Milan Night": "https://dpbossattamatka.com/panel-chart-record/milan-night.php",
    "Rajdhani Night": "https://dpbossattamatka.com/panel-chart-record/rajdhani-night.php",
    "Main Bazar": "https://dpbossattamatka.com/panel-chart-record/main-bazar.php"
}

CSV_PATH = "satta_data.csv"

def scrape_market(url):
    try:
        res = requests.get(url, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        rows = soup.find_all("tr")[1:]  # skip header
        data = []
        for row in rows:
            cols = row.find_all("td")
            if len(cols) >= 5:
                date = cols[0].text.strip()
                open_ = cols[1].text.strip()
                jodi = cols[2].text.strip()
                close = cols[3].text.strip()
                patti = cols[4].text.strip()
                data.append([date, open_, jodi, close, patti])
        return data
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return []

def update_data():
    all_data = []
    for market, url in MARKET_URLS.items():
        scraped = scrape_market(url)
        for row in scraped:
            date = pd.to_datetime(row[0], errors='coerce')
            if pd.isnull(date): continue
            all_data.append([market, date.strftime("%Y-%m-%d")] + row[1:])
    df = pd.DataFrame(all_data, columns=["Market", "Date", "Open", "Jodi", "Close", "Patti"])
    df.dropna(inplace=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df.to_csv(CSV_PATH, index=False)
    return df

def train_and_predict(df, market):
    df = df[df["Market"] == market].copy()
    df.sort_values("Date", inplace=True)
    df = df.tail(60)  # last 60 days

    if len(df) < 10:
        return None

    df["Open"] = pd.to_numeric(df["Open"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["Jodi"] = pd.to_numeric(df["Jodi"], errors="coerce")
    df["Patti"] = pd.to_numeric(df["Patti"], errors="coerce")

    df.dropna(inplace=True)

    features = df[["Open", "Close", "Jodi"]]
    targets = df["Patti"]

    model = RandomForestClassifier(n_estimators=100)
    model.fit(features, targets)

    last = features.iloc[-1:]
    pred_patti = model.predict(last)[0]

    return {
        "Open": int(last["Open"].values[0]),
        "Close": int(last["Close"].values[0]),
        "Jodi": int(last["Jodi"].values[0]),
        "Patti": int(pred_patti)
    }

def send_prediction(market, prediction):
    message = f"*{market} Prediction (Tomorrow)*\n"
    message += f"Open: `{prediction['Open']}`\n"
    message += f"Jodi: `{prediction['Jodi']}`\n"
    message += f"Close: `{prediction['Close']}`\n"
    message += f"Patti: `{prediction['Patti']}`"
    try:
        bot.send_message(chat_id=CHAT_ID, text=message, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        print("Telegram error:", e)

def main():
    df = update_data()
    markets = df["Market"].unique()

    for market in markets:
        pred = train_and_predict(df, market)
        if pred:
            send_prediction(market, pred)

if __name__ == "__main__":
    main()
