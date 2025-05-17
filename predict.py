import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from collections import Counter

TELEGRAM_BOT_TOKEN = "7121966371:AAEKHVrsqLRswXg64-6Nf3nid-Mbmlmmw5M"
CHAT_ID = "7621883960"
CSV_FILE = "satta_data.csv"
LEARNING_LOG = "prediction_log.csv"
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
    payload = {"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"}
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

today = datetime.today().strftime("%d/%m/%Y")
predictions = []
past_errors = pd.read_csv(LEARNING_LOG) if os.path.exists(LEARNING_LOG) else pd.DataFrame(columns=['Date', 'Market', 'Type', 'Wrong'])

for market in df['Market'].unique():
    mdf = df[df['Market'] == market].sort_values('Date').dropna().tail(60)

    open_digits = mdf['Jodi'].astype(str).str.zfill(2).str[0].astype(int)
    open_weights = Counter(open_digits)
    close_digits = mdf['Jodi'].astype(str).str.zfill(2).str[1].astype(int)
    close_weights = Counter(close_digits)

    # Self-adjusting weights
    for _, row in past_errors[(past_errors['Market'] == market)].iterrows():
        if row['Type'] == 'Open' and row['Wrong'] in open_weights:
            open_weights[row['Wrong']] -= 1
        if row['Type'] == 'Close' and row['Wrong'] in close_weights:
            close_weights[row['Wrong']] -= 1

    open_common = [str(d) for d, _ in open_weights.most_common(2)]
    close_common = [str(d) for d, _ in close_weights.most_common(2)]

    jodis = mdf['Jodi'].astype(str).str.zfill(2)
    jodi_common = [j for j, _ in Counter(jodis).most_common(10)]

    pattis = [o + j[0] + c for o, j, c in zip(mdf['Open'].astype(str), mdf['Jodi'].astype(str).str.zfill(2), mdf['Close'].astype(str)) if len(o)==1 and len(c)==1]
    patti_common = [p for p in pattis if len(p)==3][:4]

    message = f"""*{market.upper()}*
*Open :* {', '.join(open_common)}
*Close :* {', '.join(close_common)}
*Jodi :* {', '.join(jodi_common)}
*Patti :* {', '.join(patti_common)}"""
    predictions.append(message)
    send_telegram_message(message)

    # Log predictions (actual comparison can be added next day)
    for val in open_common:
        past_errors.loc[len(past_errors)] = [today, market, 'Open', val]
    for val in close_common:
        past_errors.loc[len(past_errors)] = [today, market, 'Close', val]

past_errors.to_csv(LEARNING_LOG, index=False)

with open("daily_prediction.txt", "w") as f:
    f.write("\n\n".join(predictions))

print("Prediction complete and saved.")
    
