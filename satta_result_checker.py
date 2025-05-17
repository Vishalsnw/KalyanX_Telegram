import os
import json
import requests
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup

# Constants
CSV_FILE = "satta_data.csv"
PRED_FILE = "today_predictions.csv"
ACCURACY_LOG = "accuracy_log.csv"
MODEL_TRACK = "model_performance.json"

TELEGRAM_BOT_TOKEN = "7121966371:AAEKHVrsqLRswXg64-6Nf3nid-Mbmlmmw5M"
CHAT_ID = "7621883960"
HEADERS = {"User-Agent": "Mozilla/5.0"}

MARKETS = {
    "Kalyan": "https://dpbossattamatka.com/panel-chart-record/kalyan.php",
    "Main Bazar": "https://dpbossattamatka.com/panel-chart-record/main-bazar.php",
    "Time Bazar": "https://dpbossattamatka.com/panel-chart-record/time-bazar.php",
    "Milan Day": "https://dpbossattamatka.com/panel-chart-record/milan-day.php",
    "Rajdhani Day": "https://dpbossattamatka.com/panel-chart-record/rajdhani-day.php",
    "Milan Night": "https://dpbossattamatka.com/panel-chart-record/milan-night.php",
    "Rajdhani Night": "https://dpbossattamatka.com/panel-chart-record/rajdhani-night.php"
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
                        date = (base_date + pd.Timedelta(days=i)).strftime("%d/%m/%Y")
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

# Load previous prediction
try:
    pred_df = pd.read_csv(PRED_FILE)
except:
    print("Prediction file not found.")
    exit()

# Parse today's actual results
today_str = datetime.today().strftime("%d/%m/%Y")
actual_data = []

for market, url in MARKETS.items():
    records = parse_table(url)
    for r in records:
        if r['Date'] == today_str:
            r['Market'] = market
            actual_data.append(r)

if not actual_data:
    print("No results found yet.")
    exit()

actual_df = pd.DataFrame(actual_data)
matched = []

for _, row in pred_df.iterrows():
    market = row['Market']
    pred_jodi = str(row['Jodi']).zfill(2)
    actual = actual_df[actual_df['Market'] == market]
    if actual.empty:
        continue
    actual_row = actual.iloc[0]
    open_match = row['Open'] == actual_row['Open']
    close_match = row['Close'] == actual_row['Close']
    jodi_match = pred_jodi == str(actual_row['Jodi']).zfill(2)
    patti_match = any(p in row['Patti'] for p in [actual_row['Open'] + actual_row['Jodi'][0] + actual_row['Close']])
    matched.append({
        "Market": market,
        "Date": today_str,
        "Open_Pred": row['Open'],
        "Open_Act": actual_row['Open'],
        "Close_Pred": row['Close'],
        "Close_Act": actual_row['Close'],
        "Jodi_Pred": pred_jodi,
        "Jodi_Act": actual_row['Jodi'],
        "Open_Match": open_match,
        "Close_Match": close_match,
        "Jodi_Match": jodi_match,
        "Patti_Match": patti_match,
        "Model": row['Model']
    })

log_df = pd.DataFrame(matched)
log_df.to_csv(ACCURACY_LOG, mode='a', header=not os.path.exists(ACCURACY_LOG), index=False)

# Telegram summary
summary = "\n\n".join(
    f"<b>{row['Market']}</b>\n<b>Open:</b> {row['Open_Pred']} vs {row['Open_Act']} ({'✔' if row['Open_Match'] else '✘'})\n"
    f"<b>Close:</b> {row['Close_Pred']} vs {row['Close_Act']} ({'✔' if row['Close_Match'] else '✘'})\n"
    f"<b>Jodi:</b> {row['Jodi_Pred']} vs {row['Jodi_Act']} ({'✔' if row['Jodi_Match'] else '✘'})\n"
    f"<b>Patti Match:</b> {'✔' if row['Patti_Match'] else '✘'}"
    for _, row in log_df.iterrows()
)

send_telegram_message("<b>Today's Prediction Accuracy:</b>\n\n" + summary)
print("Results checked and summary sent.")
