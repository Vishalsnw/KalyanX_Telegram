import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from telegram import Bot, ParseMode
import os

TOKEN = "7121966371:AAEKHVrsqLRswXg64-6Nf3nid-Mbmlmmw5M"
CHAT_ID = "7621883960"
CSV_FILE = "satta_data.csv"

MARKET_URLS = {
    "Time Bazar": "https://dpbossattamatka.com/panel-chart-record/time-bazar.php",
    "Milan Day": "https://dpbossattamatka.com/panel-chart-record/milan-day.php",
    "Rajdhani Day": "https://dpbossattamatka.com/panel-chart-record/rajdhani-day.php",
    "Kalyan": "https://dpbossattamatka.com/panel-chart-record/kalyan.php",
    "Milan Night": "https://dpbossattamatka.com/panel-chart-record/milan-night.php",
    "Rajdhani Night": "https://dpbossattamatka.com/panel-chart-record/rajdhani-night.php",
    "Main Bazar": "https://dpbossattamatka.com/panel-chart-record/main-bazar.php"
}

bot = Bot(token=TOKEN)

# Load or create CSV
if os.path.exists(CSV_FILE):
    df = pd.read_csv(CSV_FILE)
else:
    df = pd.DataFrame(columns=["Market", "Date", "Jodi", "Open", "Close", "Patti", "Prediction", "Match", "Posted"])

# Ensure Date column is datetime
df["Date"] = pd.to_datetime(df["Date"], errors='coerce')

# Scrape function
def scrape_results():
    all_rows = []
    for market, url in MARKET_URLS.items():
        try:
            soup = BeautifulSoup(requests.get(url).text, "html.parser")
            table = soup.find("table")
            rows = table.find_all("tr")[1:8]
            for r in rows:
                cols = [c.text.strip() for c in r.find_all("td")]
                if len(cols) >= 3:
                    date = pd.to_datetime(cols[0], dayfirst=True, errors='coerce')
                    if pd.isna(date): continue
                    if not ((df["Market"] == market) & (df["Date"] == date)).any():
                        open_val, close_val = ("", "")
                        if "-" in cols[2]:
                            parts = cols[2].split("-")
                            if len(parts) == 2:
                                open_val, close_val = parts
                        all_rows.append({
                            "Market": market,
                            "Date": date,
                            "Jodi": cols[1],
                            "Open": open_val,
                            "Close": close_val,
                            "Patti": cols[3] if len(cols) > 3 else "",
                            "Prediction": "",
                            "Match": "",
                            "Posted": ""
                        })
        except Exception as e:
            print(f"Error in {market}: {e}")
    return all_rows

# Add new data
new_data = scrape_results()
if new_data:
    df = pd.concat([df, pd.DataFrame(new_data)], ignore_index=True)

# Predict for tomorrow
preds = []
tomorrow = datetime.now() + timedelta(days=1)
for market in df["Market"].unique():
    market_df = df[df["Market"] == market].dropna(subset=["Jodi", "Date"])
    market_df["Date"] = pd.to_datetime(market_df["Date"], errors='coerce')
    market_df["DayOfWeek"] = market_df["Date"].dt.dayofweek
    if len(market_df) < 15: continue
    le = LabelEncoder()
    try:
        market_df["Label"] = le.fit_transform(market_df["Jodi"].astype(str))
        model = RandomForestClassifier()
        model.fit(market_df[["DayOfWeek"]], market_df["Label"])
        tomorrow_dow = tomorrow.weekday()
        pred_label = model.predict([[tomorrow_dow]])[0]
        pred_jodi = le.inverse_transform([pred_label])[0]

        existing = df[(df["Market"] == market) & (df["Date"].dt.date == tomorrow.date())]
        if not existing.empty:
            df.loc[existing.index, "Prediction"] = pred_jodi
        else:
            df = pd.concat([df, pd.DataFrame([{
                "Market": market,
                "Date": tomorrow,
                "Jodi": "",
                "Open": "",
                "Close": "",
                "Patti": "",
                "Prediction": pred_jodi,
                "Match": "",
                "Posted": ""
            }])])
        preds.append(f"*{market}*\nPrediction: *{pred_jodi}*")
    except Exception as e:
        print(f"Prediction failed for {market}: {e}")

# Send Telegram message
if preds:
    msg = "*Tomorrow's Satta Predictions:*\n\n" + "\n\n".join(preds)
    bot.send_message(chat_id=CHAT_ID, text=msg, parse_mode=ParseMode.MARKDOWN)

# Save file
df.to_csv(CSV_FILE, index=False)
