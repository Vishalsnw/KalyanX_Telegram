import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import os

# Telegram settings
TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
CHAT_ID = "YOUR_CHAT_ID"

def send_telegram_message(text):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"}
    try:
        requests.post(url, data=data, timeout=10)
    except Exception as e:
        print("Telegram error:", e)

CSV_FILE = "satta_data_with_predictions.csv"
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

today = datetime.today().date()
tomorrow = today + timedelta(days=1)
today_str = today.strftime("%d/%m/%Y")
tomorrow_str = tomorrow.strftime("%d/%m/%Y")

# Load or create CSV
if os.path.exists(CSV_FILE):
    df = pd.read_csv(CSV_FILE)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
else:
    df = pd.DataFrame(columns=['Date', 'Market', 'Open', 'Jodi', 'Close'])

# Ensure required columns
for col in ['Predicted', 'Matched', 'Posted', 'PostedAll']:
    if col not in df.columns:
        df[col] = 'No'

# Scraping
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
            for row in table.find_all("tr"):
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

existing = set(zip(df['Date'].dt.strftime('%d/%m/%Y'), df['Market']))
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
                'Close': record['Close'],
                'Predicted': 'No',
                'Matched': 'No',
                'Posted': 'No',
                'PostedAll': 'No'
            })

if new_rows:
    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)

# Add features
df['open_sum'] = df['Open'].apply(lambda x: sum(int(d) for d in str(x) if d.isdigit()) % 10)
df['close_sum'] = df['Close'].apply(lambda x: sum(int(d) for d in str(x) if d.isdigit()) % 10)
df['jodi_first'] = df['Jodi'].astype(str).str[0].astype(int, errors='ignore')
df['jodi_second'] = df['Jodi'].astype(str).str[1].astype(int, errors='ignore')
df['reverse_jodi'] = df['Jodi'].astype(str).str.zfill(2).apply(lambda x: x[::-1])
df['mirror_first'] = df['jodi_first'].apply(lambda d: (d + 5) % 10 if pd.notna(d) else d)
df['mirror_second'] = df['jodi_second'].apply(lambda d: (d + 5) % 10 if pd.notna(d) else d)
df['day_of_week'] = df['Date'].dt.day_name()
df['day_label'] = LabelEncoder().fit_transform(df['day_of_week'])
df['is_weekend'] = df['day_of_week'].isin(['Saturday', 'Sunday']).astype(int)

features = ['open_sum', 'close_sum', 'jodi_first', 'jodi_second',
            'mirror_first', 'mirror_second', 'day_label', 'is_weekend']

# Per-market predictions
results = []

for market in df['Market'].unique():
    mdf = df[df['Market'] == market].sort_values('Date')

    today_data = mdf[mdf['Date'].dt.date == today]
    if today_data[['Open', 'Jodi', 'Close']].dropna().empty:
        continue

    # Skip if already predicted for tomorrow
    tomorrow_data = mdf[(mdf['Date'].dt.date == tomorrow) & (mdf['Predicted'] == 'Yes')]
    if not tomorrow_data.empty:
        continue

    # Filter & prepare training
    mdf = mdf.dropna(subset=features + ['Jodi'])
    if len(mdf) < 60:
        continue
    mdf = mdf.tail(60)
    X = mdf[features]
    y = mdf['Jodi'].astype(str).str.zfill(2)

    model = RandomForestClassifier(n_estimators=100, max_depth=7, min_samples_split=4, random_state=42)
    model.fit(X, y)

    latest = mdf.iloc[-1]
    test = latest[features].values.reshape(1, -1)

    probs = model.predict_proba(test)[0]
    candidates = model.classes_
    top_indices = np.argsort(probs)[-10:][::-1]
    top_jodis = [candidates[i] for i in top_indices]

    digits = [d for j in top_jodis for d in j]
    digit_counts = Counter(digits)
    pred_open = [d[0] for d in digit_counts.most_common(2)]
    pred_close = [d[0] for d in digit_counts.most_common(4)[2:4]]
    pattis = [j[0] + j[1] + d for j in top_jodis[:4] for d in "0123456789"]
    pred_patti = pattis[:4]

    results.append({
        'Date': tomorrow_str,
        'Market': market,
        'Open': ', '.join(pred_open),
        'Close': ', '.join(pred_close),
        'Jodi': ', '.join(top_jodis),
        'Patti': ', '.join(pred_patti),
        'Predicted': 'Yes',
        'Matched': 'No',
        'Posted': 'No',
        'PostedAll': 'No'
    })

    if not df[(df['Date'].dt.date == tomorrow) & (df['Market'] == market) & (df['Posted'] == 'Yes')].any().any():
        msg = (
            f"*{market}*\n"
            f"*{tomorrow_str}*\n"
            f"*Open:* {', '.join(pred_open)}\n"
            f"*Close:* {', '.join(pred_close)}\n"
            f"*Jodi:* {', '.join(top_jodis)}\n"
            f"*Patti:* {', '.join(pred_patti)}"
        )
        send_telegram_message(msg)
        df.loc[(df['Date'].dt.date == tomorrow) & (df['Market'] == market), 'Posted'] = 'Yes'

# Update all predictions
df = pd.concat([df, pd.DataFrame(results)], ignore_index=True)
df.to_csv(CSV_FILE, index=False)

# 12 AM bulk posting (IST)
now = datetime.utcnow() + timedelta(hours=5, minutes=30)
if now.hour == 0:
    to_post = df[(df['Date'].dt.date == tomorrow) & (df['PostedAll'] == 'No') & (df['Predicted'] == 'Yes')]
    if not to_post.empty:
        full_msg = "*Predictions for all markets:*\n\n"
        for _, row in to_post.iterrows():
            full_msg += (
                f"*{row['Market']}*\n"
                f"*Open:* {row['Open']}\n"
                f"*Close:* {row['Close']}\n"
                f"*Jodi:* {row['Jodi']}\n"
                f"*Patti:* {row['Patti']}\n\n"
            )
        send_telegram_message(full_msg)
        df.loc[(df['Date'].dt.date == tomorrow), 'PostedAll'] = 'Yes'
        df.to_csv(CSV_FILE, index=False)
