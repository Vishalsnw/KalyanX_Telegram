import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timedelta
import os

# --- Configuration ---
TELEGRAM_BOT_TOKEN = "8050429062:AAHfDWm42MvsfxVMOcOprH6vFxcisZJqOOg"  # ✅ Updated token
CHAT_ID = "-1002573892631"

CSV_FILE = "satta_data.csv"
PRED_FILE = "today_ml_prediction.csv"
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
    payload = {"chat_id": CHAT_ID, "text": msg, "parse_mode": "Markdown"}
    try:
        r = requests.post(url, data=payload)
        print("Telegram response:", r.status_code, r.text)
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

# --- Load existing results ---
try:
    df = pd.read_csv(CSV_FILE)
    existing = set(zip(df['Date'], df['Market']))
except:
    df = pd.DataFrame(columns=['Date', 'Market', 'Open', 'Jodi', 'Close'])
    existing = set()

# --- Load message log ---
try:
    sent_log = pd.read_csv(SENT_MSG_FILE)
    sent_set = set(zip(sent_log['Date'], sent_log['Market']))
except:
    sent_log = pd.DataFrame(columns=['Date', 'Market'])
    sent_set = set()

# --- Scrape today's results ---
new_rows = []
for market, url in MARKETS.items():
    result = get_latest_result(url)
    if result.get("status") == "ok" and (result['date'], market) not in existing:
        new_rows.append({
            'Date': result['date'],
            'Market': market,
            'Open': result['open'],
            'Jodi': result['jodi'],
            'Close': result['close']
        })

if new_rows:
    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)

# --- Load Predictions ---
if os.path.exists(PRED_FILE):
    pred_df = pd.read_csv(PRED_FILE)
    pred_df['Date'] = pd.to_datetime(pred_df['Date'], dayfirst=True, errors='coerce')
    today_dt = pd.to_datetime(datetime.now().strftime("%d/%m/%Y"))
    pred_df = pred_df[pred_df['Date'] == today_dt]
    pred_df['Date'] = pred_df['Date'].dt.strftime("%d/%m/%Y")
else:
    pred_df = pd.DataFrame()

# --- Match with today's actuals ---
today = datetime.now().strftime("%d/%m/%Y")
today_actuals = df[df['Date'] == today]

matched_msgs = []
unmatched_msgs = []

def emoji(b): return "✅" if b else "❌"

def format_card(market, open_act, close_act, jodi_act, pred=None):
    if pred is not None:
        pred_open = [x.strip() for x in str(pred.get("Open", "")).split(",")]
        pred_close = [x.strip() for x in str(pred.get("Close", "")).split(",")]
        pred_jodi = [x.strip().zfill(2) for x in str(pred.get("Jodis", "")).split(",")]
        pred_patti = [x.strip() for x in str(pred.get("Pattis", "")).split(",")]
        act_pattis = [x.strip() for x in str(pred.get("Patti", "")).split(",") if x.strip()]

        open_match = jodi_act[0] in pred_open
        close_match = jodi_act[1] in pred_close
        jodi_match = jodi_act in pred_jodi
        patti_match = any(p in pred_patti for p in act_pattis)

        return (
            f"\n📍 *{market}*\n"
            f"*Open:* {', '.join(pred_open)} vs `{jodi_act[0]}` {emoji(open_match)}\n"
            f"*Close:* {', '.join(pred_close)} vs `{jodi_act[1]}` {emoji(close_match)}\n"
            f"*Jodi:* {', '.join(pred_jodi)} vs `{jodi_act}` {emoji(jodi_match)}\n"
            f"*Patti:* {emoji(patti_match)}"
        )
    else:
        return (
            f"\n📍 *{market}*\n"
            f"*Open:* `{open_act}`\n"
            f"*Close:* `{close_act}`\n"
            f"*Jodi:* `{jodi_act}`"
        )

# --- Message Creation Loop ---
for _, row in today_actuals.iterrows():
    market = row["Market"]
    if (today, market) in sent_set:
        continue

    ao, aj, ac = str(row["Open"]), str(row["Jodi"]), str(row["Close"])
    pred_row = pred_df[pred_df["Market"] == market]
    pred = pred_row.iloc[0] if not pred_row.empty else None

    if pred is not None:
        pred_open = [x.strip() for x in str(pred.get("Open", "")).split(",")]
        pred_close = [x.strip() for x in str(pred.get("Close", "")).split(",")]
        pred_jodi = [x.strip().zfill(2) for x in str(pred.get("Jodis", "")).split(",")]
        pred_patti = [x.strip() for x in str(pred.get("Pattis", "")).split(",")]
        act_pattis = [x.strip() for x in str(pred.get("Patti", "")).split(",") if x.strip()]

        open_match = aj[0] in pred_open
        close_match = aj[1] in pred_close
        jodi_match = aj in pred_jodi
        patti_match = any(p in pred_patti for p in act_pattis)

        if open_match or close_match or jodi_match or patti_match:
            msg = format_card(market, ao, ac, aj, pred=pred)
            matched_msgs.append(msg)
    else:
        msg = format_card(market, ao, ac, aj, pred=None)
        unmatched_msgs.append(msg)

    sent_log = pd.concat([sent_log, pd.DataFrame([{"Date": today, "Market": market}])], ignore_index=True)

# --- Send Messages ---
if matched_msgs:
    send_telegram_message("*🌟 Prediction Match Found*\n" + "\n".join(matched_msgs))
elif unmatched_msgs:
    send_telegram_message("*📊 Today's Results (No Match)*\n" + "\n".join(unmatched_msgs))
else:
    print("Nothing new to send.")

# --- Save Sent Log ---
sent_log.to_csv(SENT_MSG_FILE, index=False)
print("Script finished.")
