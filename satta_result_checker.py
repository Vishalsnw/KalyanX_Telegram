import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import os

# --- Configuration ---
TELEGRAM_BOT_TOKEN = "8050429062:AAGjX5t7poexZWjIEuMijQ1bVOJELqgdlmc"
CHAT_ID = "-1002573892631"

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

# --- Load Data ---
try:
    df = pd.read_csv(CSV_FILE)
    existing = set(zip(df['Date'], df['Market']))
except:
    df = pd.DataFrame(columns=['Date', 'Market', 'Open', 'Jodi', 'Close'])
    existing = set()

try:
    sent_log = pd.read_csv(SENT_MSG_FILE)
    sent_set = set(zip(sent_log['Date'], sent_log['Market']))
except:
    sent_log = pd.DataFrame(columns=['Date', 'Market'])
    sent_set = set()

# --- Scrape and Append New Results ---
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

# --- Load Prediction File ---
if not os.path.exists(PRED_FILE):
    print("Prediction file not found. Sending actuals only.")
    send_actuals_only = True
    pred_df = pd.DataFrame()
else:
    pred_df = pd.read_csv(PRED_FILE)
    if 'Date' in pred_df.columns:
        pred_df['Date'] = pd.to_datetime(pred_df['Date'], errors='coerce').dt.strftime("%d/%m/%Y")
        latest_pred_date = pred_df['Date'].dropna().max()
        pred_df = pred_df[pred_df['Date'] == latest_pred_date]
    else:
        pred_df = pd.DataFrame()
    send_actuals_only = pred_df.empty

# --- Prepare for Matching ---
df["Date"] = df["Date"].astype(str)
today = datetime.now().strftime("%d/%m/%Y")
today_actuals = df[df["Date"] == today]

matched = []
messages = []
unmatched_msgs = []

def emoji_match(is_match):
    return '✅' if is_match else '❌'

for _, row in today_actuals.iterrows():
    market = row["Market"]
    if (today, market) in sent_set:
        continue  # Skip already sent
    ao, aj, ac = str(row["Open"]), str(row["Jodi"]), str(row["Close"])
    if not ao or not aj or not ac or len(aj) != 2:
        continue

    actual_message = (
        f"<b>{market}</b>\n"
        f"<b>Open:</b> {ao}\n"
        f"<b>Close:</b> {ac}\n"
        f"<b>Jodi:</b> {aj}"
    )

    pred_row = pred_df[pred_df["Market"] == market]
    if not pred_row.empty:
        pr = pred_row.iloc[0]
        pred_open = [x.strip() for x in str(pr.get("Open", "")).split(",")]
        pred_close = [x.strip() for x in str(pr.get("Close", "")).split(",")]
        pred_jodi = [x.strip().zfill(2) for x in str(pr.get("Jodis", "")).split(",")]
        pred_patti = [x.strip() for x in str(pr.get("Pattis", "")).split(",")]
        actual_pattis = [x.strip() for x in str(row.get("Patti", "")).split(",") if x.strip()]

        open_match = aj[0] in pred_open
        close_match = aj[1] in pred_close
        jodi_match = aj in pred_jodi
        patti_match = any(p in pred_patti for p in actual_pattis)

        if any([open_match, close_match, jodi_match, patti_match]):
            msg = (
                f"<b>{market}</b>\n"
                f"<b>Open:</b> {', '.join(pred_open)} vs {aj[0]} {emoji_match(open_match)}\n"
                f"<b>Close:</b> {', '.join(pred_close)} vs {aj[1]} {emoji_match(close_match)}\n"
                f"<b>Jodi:</b> {', '.join(pred_jodi)} vs {aj} {emoji_match(jodi_match)}\n"
                f"<b>Patti:</b> {emoji_match(patti_match)}"
            )
            messages.append(msg)
            matched.append({
                "Market": market, "Date": today,
                "Open_Pred": ','.join(pred_open), "Open_Act": aj[0],
                "Close_Pred": ','.join(pred_close), "Close_Act": aj[1],
                "Jodi_Pred": ','.join(pred_jodi), "Jodi_Act": aj,
                "Open_Match": open_match, "Close_Match": close_match,
                "Jodi_Match": jodi_match, "Patti_Match": patti_match,
                "Model": pr.get("Model", "N/A")
            })
            sent_log = pd.concat([sent_log, pd.DataFrame([{'Date': today, 'Market': market}])], ignore_index=True)
        else:
            unmatched_msgs.append(actual_message)
            sent_log = pd.concat([sent_log, pd.DataFrame([{'Date': today, 'Market': market}])], ignore_index=True)
    else:
        unmatched_msgs.append(actual_message)
        sent_log = pd.concat([sent_log, pd.DataFrame([{'Date': today, 'Market': market}])], ignore_index=True)

# --- Send Messages ---
if matched:
    pd.DataFrame(matched).to_csv(ACCURACY_LOG, mode='a', header=not os.path.exists(ACCURACY_LOG), index=False)
    send_telegram_message("<b>🎯 Match Found</b>\n\n" + "\n\n".join(messages))

if unmatched_msgs:
    send_telegram_message("<b>📊 Today's Results</b>\n\n" + "\n\n".join(unmatched_msgs))

# --- Save Sent Log ---
sent_log.to_csv(SENT_MSG_FILE, index=False)
print("📨 Telegram messages sent.")
