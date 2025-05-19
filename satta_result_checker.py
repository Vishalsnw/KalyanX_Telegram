import os
import pandas as pd
from datetime import datetime
import requests

# Constants
CSV_FILE = "satta_data.csv"
PRED_FILE = "today_predictions.csv"
ACCURACY_LOG = "accuracy_log.csv"

TELEGRAM_BOT_TOKEN = "7121966371:AAEKHVrsqLRswXg64-6Nf3nid-Mbmlmmw5M"
CHAT_ID = "7621883960"

def send_telegram_message(msg):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print("Telegram error:", e)

# Load data
if not os.path.exists(CSV_FILE):
    print("satta_data.csv not found.")
    exit()

df = pd.read_csv(CSV_FILE)
df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")

today = datetime.now().strftime("%d/%m/%Y")
today_dt = datetime.strptime(today, "%d/%m/%Y")

# Load today's actual results from CSV
today_actuals = df[df["Date"] == today_dt]

if not os.path.exists(PRED_FILE):
    print("Prediction file not found. Skipping accuracy check.")
    exit()

try:
    pred_df = pd.read_csv(PRED_FILE)
except Exception as e:
    print("Error reading prediction file:", e)
    exit()

matched = []

for _, row in pred_df.iterrows():
    market = row['Market']
    pred_jodi = str(row['Jodi']).zfill(2)

    actual = today_actuals[today_actuals['Market'] == market]
    if actual.empty:
        matched.append({
            "Market": market,
            "Date": today,
            "Open_Pred": row['Open'],
            "Open_Act": "Pending",
            "Close_Pred": row['Close'],
            "Close_Act": "Pending",
            "Jodi_Pred": pred_jodi,
            "Jodi_Act": "Pending",
            "Open_Match": "Pending",
            "Close_Match": "Pending",
            "Jodi_Match": "Pending",
            "Patti_Match": "Pending",
            "Model": row.get('Model', 'N/A')
        })
        continue

    actual_row = actual.iloc[0]
    open_match = str(row['Open']) == str(actual_row['Open'])
    close_match = str(row['Close']) == str(actual_row['Close'])
    jodi_match = pred_jodi == str(actual_row['Jodi']).zfill(2)
    predicted_pattis = [p.strip() for p in str(row.get('Patti', '')).split(',')]
    full_patti = actual_row['Open'] + actual_row['Jodi'][0] + actual_row['Close']
    patti_match = full_patti in predicted_pattis

    matched.append({
        "Market": market,
        "Date": today,
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
        "Model": row.get('Model', 'N/A')
    })

# Save to accuracy log
log_df = pd.DataFrame(matched)
log_df.to_csv(ACCURACY_LOG, mode='a', header=not os.path.exists(ACCURACY_LOG), index=False)

# Send Telegram Summary
summary = []
for row in matched:
    summary.append(
        f"<b>{row['Market']}</b>\n"
        f"<b>Open:</b> {row['Open_Pred']} vs {row['Open_Act']} ({'✔' if row['Open_Match'] == True else '✘' if row['Open_Match'] == False else 'Pending'})\n"
        f"<b>Close:</b> {row['Close_Pred']} vs {row['Close_Act']} ({'✔' if row['Close_Match'] == True else '✘' if row['Close_Match'] == False else 'Pending'})\n"
        f"<b>Jodi:</b> {row['Jodi_Pred']} vs {row['Jodi_Act']} ({'✔' if row['Jodi_Match'] == True else '✘' if row['Jodi_Match'] == False else 'Pending'})\n"
        f"<b>Patti Match:</b> {'✔' if row['Patti_Match'] == True else '✘' if row['Patti_Match'] == False else 'Pending'}"
    )

final_message = "<b>Today's Prediction Accuracy:</b>\n\n" + "\n\n".join(summary)
send_telegram_message(final_message)
print("Telegram summary sent.")
