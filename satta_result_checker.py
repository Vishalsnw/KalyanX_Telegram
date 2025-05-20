import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import os

# Constants and Files
CSV_FILE = "satta_data.csv"
PRED_FILE = "today_ml_prediction.csv"
ACCURACY_LOG = "accuracy_log.csv"
HEADERS = {"User-Agent": "Mozilla/5.0"}
TELEGRAM_BOT_TOKEN = "7121966371:AAEKHVrsqLRswXg64-6Nf3nid-Mbmlmmw5M"
CHAT_ID = "7621883960"

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

def get_latest_result(url):
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        res.encoding = 'utf-8'
        soup = BeautifulSoup(res.text, 'html.parser')
        for table in soup.find_all("table"):
            rows = table.find_all("tr")
            for row in reversed(rows):
                cols = row.find_all("td")
                if len(cols) >= 4 and 'to' in cols[0].text:
                    start_date = cols[0].text.split('to')[0].strip()
                    try:
                        base_date = datetime.strptime(start_date, "%d/%m/%Y")
                    except:
                        continue
                    cells = cols[1:]
                    index = len(cells) // 3 - 1
                    date = (base_date + timedelta(days=index)).strftime("%d/%m/%Y")
                    o, j, c = cells[index*3: index*3+3]
                    if '**' in o.text or '**' in j.text or '**' in c.text:
                        return {'date': date, 'open': '', 'jodi': '', 'close': '', 'status': 'Not declared'}
                    return {
                        'date': date,
                        'open': parse_cell(o),
                        'jodi': parse_cell(j),
                        'close': parse_cell(c),
                        'status': 'ok'
                    }
    except Exception as e:
        return {'status': f'error: {e}'}

# Load existing results
try:
    df = pd.read_csv(CSV_FILE)
    existing = set(zip(df['Date'], df['Market']))
except:
    df = pd.DataFrame(columns=['Date', 'Market', 'Open', 'Jodi', 'Close'])
    existing = set()

new_rows = []

# Scrape latest results
for market, url in MARKETS.items():
    print(f"Checking {market}...")
    result = get_latest_result(url)
    if result.get("status") == "ok":
        if (result['date'], market) not in existing:
            print(f"  ➕ New result found: {result['date']} - {market}")
            new_rows.append({
                'Date': result['date'],
                'Market': market,
                'Open': result['open'],
                'Jodi': result['jodi'],
                'Close': result['close']
            })
        else:
            print(f"  ✅ Already in CSV: {result['date']} - {market}")
    else:
        print(f"  ⚠️ Skipped {market}: {result.get('status')}")

# Append new results
if new_rows:
    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)
    print(f"\n✅ Appended {len(new_rows)} new rows to {CSV_FILE}")
else:
    print("\n✅ No new results found")

# Prediction match
if not os.path.exists(PRED_FILE):
    print("Prediction file not found. Skipping match check.")
    exit()

df["Date"] = df["Date"].astype(str)
pred_df = pd.read_csv(PRED_FILE)
today = datetime.now().strftime("%d/%m/%Y")
today_actuals = df[df["Date"] == today]

matched = []
messages = []

for _, row in pred_df.iterrows():
    market = row['Market']
    pred_open = [x.strip() for x in str(row.get('Open', '')).split(',')]
    pred_close = [x.strip() for x in str(row.get('Close', '')).split(',')]
    pred_jodi = [x.strip().zfill(2) for x in str(row.get('Jodis', '')).split(',')]
    pred_patti = [x.strip() for x in str(row.get('Pattis', '')).split(',')]

    actual = today_actuals[today_actuals['Market'] == market]
    if actual.empty:
        continue

    actual_row = actual.iloc[0]
    ao, ac, aj = str(actual_row['Open']), str(actual_row['Close']), str(actual_row['Jodi'])

    if not ao or not ac or not aj or len(aj) < 1:
        print(f"Skipping {market}: Incomplete actual data -> Open: {ao}, Jodi: {aj}, Close: {ac}")
        continue

    ap = ao + aj[0] + ac
    open_match = ao in pred_open
    close_match = ac in pred_close
    jodi_match = aj in pred_jodi
    patti_match = ap in pred_patti

    matched.append({
        "Market": market, "Date": today,
        "Open_Pred": ','.join(pred_open), "Open_Act": ao,
        "Close_Pred": ','.join(pred_close), "Close_Act": ac,
        "Jodi_Pred": ','.join(pred_jodi), "Jodi_Act": aj,
        "Open_Match": open_match, "Close_Match": close_match,
        "Jodi_Match": jodi_match, "Patti_Match": patti_match,
        "Model": row.get('Model', 'N/A')
    })

    messages.append(
        f"<b>{market}</b>\n"
        f"<b>Open:</b> {','.join(pred_open)} vs {ao} ({'✔' if open_match else '✘'})\n"
        f"<b>Close:</b> {','.join(pred_close)} vs {ac} ({'✔' if close_match else '✘'})\n"
        f"<b>Jodi:</b> {','.join(pred_jodi)} vs {aj} ({'✔' if jodi_match else '✘'})\n"
        f"<b>Patti Match:</b> {'✔' if patti_match else '✘'}"
    )

if matched:
    pd.DataFrame(matched).to_csv(ACCURACY_LOG, mode='a', header=not os.path.exists(ACCURACY_LOG), index=False)

if messages:
    full_msg = "<b>Market Result Matched:</b>\n\n" + "\n\n".join(messages)
    send_telegram_message(full_msg)
    print("Telegram message sent.")
else:
    print("No new market result available yet.")
