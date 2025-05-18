import os
import json
import requests
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup

# Constants
CSV_FILE = "enhanced_satta_data.csv"
PRED_FILE = "today_predictions.csv"
ACCURACY_LOG = "accuracy_log.csv"

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

def parse_table(url, market):
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
                            'Market': market,
                            'Open': parse_cell(o),
                            'Jodi': parse_cell(j),
                            'Close': parse_cell(c)
                        })
        return results
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return []

def enrich_data(df):
    df["day_of_week"] = pd.to_datetime(df["Date"], format="%d/%m/%Y").dt.day_name()
    df["is_weekend"] = df["day_of_week"].isin(["Saturday", "Sunday"])
    df["open_sum"] = df["Open"].apply(lambda x: sum(int(d) for d in str(x) if d.isdigit()))
    df["close_sum"] = df["Close"].apply(lambda x: sum(int(d) for d in str(x) if d.isdigit()))
    df["mirror_open"] = df["Open"].apply(lambda x: ''.join(str((9 - int(d)) % 10) for d in x if d.isdigit()))
    df["mirror_close"] = df["Close"].apply(lambda x: ''.join(str((9 - int(d)) % 10) for d in x if d.isdigit()))
    df["reverse_jodi"] = df["Jodi"].apply(lambda x: str(x).zfill(2)[::-1])
    df["is_holiday"] = False

    # prev_jodi_distance per market
    enhanced = []
    for market in df["Market"].unique():
        mkt_df = df[df["Market"] == market].sort_values("Date").copy()
        mkt_df["prev_jodi_distance"] = mkt_df["Jodi"].astype(int).diff().abs().fillna(0).astype(int)
        enhanced.append(mkt_df)
    return pd.concat(enhanced)

# Step 1: Load existing data
existing_df = pd.read_csv(CSV_FILE) if os.path.exists(CSV_FILE) else pd.DataFrame(columns=["Date", "Market"])

# Step 2: Fetch full results from all markets
all_new_data = []

for market, url in MARKETS.items():
    raw_results = parse_table(url, market)
    for row in raw_results:
        if not ((existing_df['Date'] == row['Date']) & (existing_df['Market'] == row['Market'])).any():
            all_new_data.append(row)

if not all_new_data:
    print("No new results to update.")
else:
    print(f"Adding {len(all_new_data)} new rows to CSV.")
    new_df = pd.DataFrame(all_new_data)
    enriched_df = enrich_data(new_df)
    final_df = pd.concat([existing_df, enriched_df], ignore_index=True)
    final_df = final_df.drop_duplicates(subset=["Date", "Market"]).sort_values(["Date", "Market"])
    final_df.to_csv(CSV_FILE, index=False)
    print("Updated enhanced_satta_data.csv")

# Step 3: Accuracy check for today
today_str = datetime.today().strftime("%d/%m/%Y")
today_actuals = enriched_df[enriched_df['Date'] == today_str] if not enriched_df.empty else pd.DataFrame()

try:
    pred_df = pd.read_csv(PRED_FILE)
except:
    print("Prediction file not found.")
    exit()

matched = []

for _, row in pred_df.iterrows():
    market = row['Market']
    pred_jodi = str(row['Jodi']).zfill(2)
    actual = today_actuals[today_actuals['Market'] == market]
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
if not log_df.empty:
    summary = "\n\n".join(
        f"<b>{row['Market']}</b>\n<b>Open:</b> {row['Open_Pred']} vs {row['Open_Act']} ({'✔' if row['Open_Match'] else '✘'})\n"
        f"<b>Close:</b> {row['Close_Pred']} vs {row['Close_Act']} ({'✔' if row['Close_Match'] else '✘'})\n"
        f"<b>Jodi:</b> {row['Jodi_Pred']} vs {row['Jodi_Act']} ({'✔' if row['Jodi_Match'] else '✘'})\n"
        f"<b>Patti Match:</b> {'✔' if row['Patti_Match'] else '✘'}"
        for _, row in log_df.iterrows()
    )
    send_telegram_message("<b>Today's Prediction Accuracy:</b>\n\n" + summary)
    print("Telegram summary sent.")
else:
    print("No match data to report.")
