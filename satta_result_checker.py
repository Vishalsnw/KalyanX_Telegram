import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

CSV_FILE = "satta_data.csv"
PRED_FILE = "today_ml_prediction.csv"
ACCURACY_LOG = "accuracy_log.csv"
SENT_MSG_FILE = "sent_messages.csv"
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
        r = requests.post(url, data=payload)
        if not r.ok:
            print("Telegram send error:", r.text)
    except Exception as e:
        print("Telegram exception:", e)

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

# Load previous results
try:
    df = pd.read_csv(CSV_FILE)
    existing = set(zip(df['Date'], df['Market']))
except:
    df = pd.DataFrame(columns=['Date', 'Market', 'Open', 'Jodi', 'Close'])
    existing = set()

# Load sent log
try:
    sent_log = pd.read_csv(SENT_MSG_FILE)
    sent_set = set(zip(sent_log['Date'], sent_log['Market']))
except:
    sent_log = pd.DataFrame(columns=['Date', 'Market'])
    sent_set = set()

# Scrape and collect new results
new_rows = []
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

if new_rows:
    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)
    print(f"\n✅ Appended {len(new_rows)} new rows to {CSV_FILE}")
else:
    print("\n✅ No new results found")

# Prediction file check
if not os.path.exists(PRED_FILE):
    print("Prediction file not found. Skipping match check.")
    exit()

df["Date"] = df["Date"].astype(str)
pred_df = pd.read_csv(PRED_FILE)

# Filter prediction for latest date
if 'Date' in pred_df.columns:
    pred_df['Date'] = pd.to_datetime(pred_df['Date'], errors='coerce').dt.strftime("%d/%m/%Y")
    latest_pred_date = pred_df['Date'].dropna().max()
    pred_df = pred_df[pred_df['Date'] == latest_pred_date]
else:
    latest_pred_date = "N/A"

today = datetime.now().strftime("%d/%m/%Y")
today_actuals = df[df["Date"] == today]

matched = []
messages = []

def emoji_match(is_match):
    return '✅' if is_match else '❌'

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
    actual_patti_raw = str(actual_row.get('Patti', '')).strip()
    actual_pattis = [x.strip() for x in actual_patti_raw.split(',') if x.strip()]

    if not ao or not ac or not aj or len(aj) != 2:
        print(f"Skipping {market}: Incomplete actual data -> Open: {ao}, Jodi: {aj}, Close: {ac}")
        continue

    open_match = aj[0] in pred_open
    close_match = aj[1] in pred_close
    jodi_match = aj in pred_jodi
    patti_match = any(p in pred_patti for p in actual_pattis)

    if not any([open_match, close_match, jodi_match, patti_match]):
        continue  # skip if no match

    matched.append({
        "Market": market, "Date": today,
        "Open_Pred": ','.join(pred_open), "Open_Act": aj[0],
        "Close_Pred": ','.join(pred_close), "Close_Act": aj[1],
        "Jodi_Pred": ','.join(pred_jodi), "Jodi_Act": aj,
        "Open_Match": open_match, "Close_Match": close_match,
        "Jodi_Match": jodi_match, "Patti_Match": patti_match,
        "Model": row.get('Model', 'N/A')
    })

    if (today, market) not in sent_set:
        message = (
            f"<b>{market}</b>\n"
            f"<b>Open:</b> {', '.join(pred_open)} vs {aj[0]} {emoji_match(open_match)}\n"
            f"<b>Close:</b> {', '.join(pred_close)} vs {aj[1]} {emoji_match(close_match)}\n"
            f"<b>Jodi:</b> {', '.join(pred_jodi)} vs {aj} {emoji_match(jodi_match)}\n"
            f"<b>Patti:</b> {emoji_match(patti_match)}"
        )
        messages.append(message)
        sent_set.add((today, market))
        sent_log = pd.concat([sent_log, pd.DataFrame([{'Date': today, 'Market': market}])], ignore_index=True)

# Save matched log
if matched:
    pd.DataFrame(matched).to_csv(ACCURACY_LOG, mode='a', header=not os.path.exists(ACCURACY_LOG), index=False)

# Send Telegram message
if messages:
    full_msg = "<b>🎯 Market Match Found</b>\n\n" + "\n\n".join(messages)
    send_telegram_message(full_msg)
    sent_log.to_csv(SENT_MSG_FILE, index=False)
    print("📨 Telegram message sent.")
else:
    print("❌ No match to send.")
