import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from telegram import Bot
from telegram.constants import ParseMode
import numpy as np
import sys
import os

# Telegram config
TOKEN = "7121966371:AAEKHVrsqLRswXg64-6Nf3nid-Mbmlmmw5M"
CHAT_ID = "7621883960"
bot = Bot(token=TOKEN)

# Market URLs
MARKET_URLS = {
    "Time Bazar": "https://dpbossattamatka.com/panel-chart-record/time-bazar.php",
    "Milan Day": "https://dpbossattamatka.com/panel-chart-record/milan-day.php",
    "Rajdhani Day": "https://dpbossattamatka.com/panel-chart-record/rajdhani-day.php",
    "Kalyan": "https://dpbossattamatka.com/panel-chart-record/kalyan.php",
    "Milan Night": "https://dpbossattamatka.com/panel-chart-record/milan-night.php",
    "Rajdhani Night": "https://dpbossattamatka.com/panel-chart-record/rajdhani-night.php",
    "Main Bazar": "https://dpbossattamatka.com/panel-chart-record/main-bazar.php"
}

CSV_PATH = "enhanced_satta_data.csv"
TODAY = datetime.now().strftime("%Y-%m-%d")
TOMORROW = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

def scrape_market(url):
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        rows = soup.select("table tr")[1:]  # skip header
        print(f"[DEBUG] {url} - rows found: {len(rows)}")
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
        print(f"[ERROR] Scraping {url}: {e}")
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
    return df

def load_or_create_csv(df):
    if os.path.exists(CSV_PATH):
        old = pd.read_csv(CSV_PATH, parse_dates=["Date"])
        combined = pd.concat([old, df]).drop_duplicates(["Market", "Date"]).reset_index(drop=True)
    else:
        combined = df
    if "Posted" not in combined.columns:
        combined["Posted"] = ""
    combined.to_csv(CSV_PATH, index=False)
    return combined

def train_and_predict(df, market):
    df = df[df["Market"] == market].copy()
    df = df.sort_values("Date").tail(60)

    df["Open"] = pd.to_numeric(df["Open"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["Jodi"] = pd.to_numeric(df["Jodi"], errors="coerce")
    df["Patti"] = pd.to_numeric(df["Patti"], errors="coerce")
    df.dropna(inplace=True)

    if len(df) < 10: return None

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

def send_prediction(market, pred, summary_mode=False):
    message = f"*{market} Prediction ({TOMORROW})*\n"
    message += f"Open: `{pred['Open']}`\n"
    message += f"Jodi: `{pred['Jodi']}`\n"
    message += f"Close: `{pred['Close']}`\n"
    message += f"Patti: `{pred['Patti']}`"
    print(f"\nSending:\n{message}")
    if not summary_mode:
        try:
            bot.send_message(chat_id=CHAT_ID, text=message, parse_mode=ParseMode.MARKDOWN)
        except Exception as e:
            print("Telegram error:", e)
    return message

def mark_posted(csv_df, market):
    csv_df.loc[(csv_df["Market"] == market) & (csv_df["Date"] == TODAY), "Posted"] = "Yes"

def main():
    print("[INFO] Scraping fresh data...")
    fresh_df = update_data()
    print("[INFO] Loading combined dataset...")
    df = load_or_create_csv(fresh_df)

    is_summary = "--summary" in sys.argv
    all_messages = []

    for market in df["Market"].unique():
        latest = df[(df["Market"] == market)].sort_values("Date").tail(1)
        if latest.empty or latest["Date"].dt.strftime("%Y-%m-%d").values[0] != TODAY:
            print(f"[SKIP] {market} has no data for today.")
            continue
        if latest["Posted"].values[0] == "Yes" and not is_summary:
            print(f"[SKIP] {market} already posted.")
            continue

        pred = train_and_predict(df, market)
        if pred:
            msg = send_prediction(market, pred, summary_mode=is_summary)
            all_messages.append(msg)
            if not is_summary:
                mark_posted(df, market)

    if is_summary and all_messages:
        final = "\n\n".join(all_messages)
        try:
            bot.send_message(chat_id=CHAT_ID, text=f"*Daily Summary ({TOMORROW})*\n\n{final}", parse_mode=ParseMode.MARKDOWN)
        except Exception as e:
            print("Summary send error:", e)

    df.to_csv(CSV_PATH, index=False)

if __name__ == "__main__":
    main()
