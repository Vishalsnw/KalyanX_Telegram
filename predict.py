# satta_predictor.py
import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
import joblib

TELEGRAM_BOT_TOKEN = "<your_token_here>"
CHAT_ID = "<your_chat_id_here>"
CSV_FILE = "satta_data.csv"
LOG_FILE = "prediction_log.csv"
HEADERS = {"User-Agent": "Mozilla/5.0"}

MARKETS = {
    "Time Bazar": "https://dpbossattamatka.com/panel-chart-record/time-bazar.php",
    "Milan Day": "https://dpbossattamatka.com/panel-chart-record/milan-day.php",
    "Rajdhani Day": "https://dpbossattamatka.com/panel-chart-record/rajdhani-day.php",
    "Kalyan": "https://dpbossattamatka.com/panel-chart-record/kalyan.php",
    "Milan Night": "https://dpbossattamatka.com/panel-chart-record/milan-night.php",
    "Rajdhani Night": "https://dpbossattamatka.com/panel-chart-record/rajdhani-night.php",
    "Main Bazar": "https://dpbossattamatka.com/panel-chart-record/main-bazar.php"
}

def send_telegram_message(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print("Telegram error:", e)

def parse_cell(cell):
    parts = cell.decode_contents().split('<br>')
    return ''.join(BeautifulSoup(p, 'html.parser').get_text(strip=True) for p in parts)

def parse_table(url):
    results = []
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        res.encoding = 'utf-8'
        soup = BeautifulSoup(res.text, 'html.parser')
        for table in soup.find_all("table"):
            rows = table.find_all("tr")
            for row in rows:
                cols = row.find_all("td")
                if len(cols) >= 4 and 'to' in cols[0].text:
                    try:
                        base_date = datetime.strptime(cols[0].text.split("to")[0].strip(), "%d/%m/%Y")
                    except:
                        continue
                    cells = cols[1:]
                    total_days = len(cells) // 3
                    for i in range(total_days):
                        date = (base_date + timedelta(days=i)).strftime("%d/%m/%Y")
                        o, j, c = cells[i*3:i*3+3]
                        if '**' in o.text or '**' in j.text or '**' in c.text:
                            continue
                        results.append({
                            'Date': date,
                            'Open': parse_cell(o),
                            'Jodi': parse_cell(j),
                            'Close': parse_cell(c)
                        })
        return results
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return []

def get_next_prediction_date():
    next_day = datetime.now() + timedelta(days=1)
    if next_day.weekday() == 6:
        next_day += timedelta(days=1)
    return next_day.strftime("%d/%m/%Y")

try:
    df = pd.read_csv(CSV_FILE)
    existing = set(zip(df['Date'], df['Market']))
except:
    df = pd.DataFrame(columns=['Date', 'Market', 'Open', 'Jodi', 'Close'])
    existing = set()

new_rows = []
for market, url in MARKETS.items():
    records = parse_table(url)
    for record in records:
        if (record['Date'], market) not in existing:
            new_rows.append({
                'Date': record['Date'],
                'Market': market,
                'Open': record['Open'],
                'Jodi': record['Jodi'],
                'Close': record['Close']
            })

if new_rows:
    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)

df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df = df.dropna(subset=['Date', 'Jodi', 'Open', 'Close'])

predictions = []
pred_date = get_next_prediction_date()

for market in df['Market'].unique():
    mdf = df[df['Market'] == market].sort_values('Date')
    mdf = mdf.tail(60)

    if len(mdf) < 10:
        continue

    # ML model input preparation
    jodis = mdf['Jodi'].astype(str).str.zfill(2)
    open_digits = jodis.str[0].astype(int)
    close_digits = jodis.str[1].astype(int)

    open_common = [str(d) for d, _ in Counter(open_digits).most_common(2)]
    close_common = [str(d) for d, _ in Counter(close_digits).most_common(2)]
    jodi_common = [j for j, _ in Counter(jodis).most_common(10)]

    pattis = [o + j[0] + c for o, j, c in zip(mdf['Open'].astype(str), jodis, mdf['Close'].astype(str)) if o.isdigit() and c.isdigit()]
    patti_common = [p for p in pattis if len(p) == 3 or len(p) == 4][-4:]

    prediction = f"""
<b>{market.upper()}</b>
<b>{pred_date}</b>
<b>Open:</b> {', '.join(open_common)}
<b>Close:</b> {', '.join(close_common)}
<b>Jodi:</b> {', '.join(jodi_common)}
<b>Patti:</b> {', '.join(patti_common)}"""

    predictions.append(prediction)
    send_telegram_message(prediction)

    # Log prediction
    log_data = pd.DataFrame([{
        'Date': pred_date,
        'Market': market,
        'Open': ','.join(open_common),
        'Close': ','.join(close_common),
        'Jodi': ','.join(jodi_common),
        'Patti': ','.join(patti_common),
        'Matched': ''
    }])
    if os.path.exists(LOG_FILE):
        log_df = pd.read_csv(LOG_FILE)
        log_df = pd.concat([log_df, log_data], ignore_index=True)
    else:
        log_df = log_data
    log_df.to_csv(LOG_FILE, index=False)

print("Predictions sent.")
