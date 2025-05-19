import os
import pandas as pd
from datetime import datetime
import requests

# Constants
CSV_FILE = "satta_data.csv"
PRED_FILE = "today_ml_prediction.csv"
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

# Check files
if not os.path.exists(CSV_FILE):
    print("satta_data.csv not found.")
    exit()

if not os.path.exists(PRED_FILE):
    print("Prediction file not found. Skipping accuracy check.")
    exit()

# Load data
df = pd.read_csv(CSV_FILE)
df["Date"] = df["Date"].astype(str)

pred_df = pd.read_csv(PRED_FILE)
today = datetime.now().strftime("%d/%m/%Y")
today_actuals = df[df["Date"] == today]

matched = []

for _, row in pred_df.iterrows():
    market = row['Market']
    predicted_opens = [x.strip() for x in str(row.get('Open', '')).split(',')]
    predicted_closes = [x.strip() for x in str(row.get('Close', '')).split(',')]
    predicted_jodis = [x.strip().zfill(2) for x in str(row.get('Jodis', '')).split(',')]
    predicted_pattis = [x.strip() for x in str(row.get('Pattis', '')).split(',')]

    actual = today_actuals[today_actuals['Market'] == market]
    if actual.empty:
        matched.append({
            "Market": market,
            "Date": today,
            "Open_Pred": ','.join(predicted_opens),
            "Open_Act": "Pending",
            "Close_Pred": ','.join(predicted_closes),
            "Close_Act": "Pending",
            "Jodi_Pred": ','.join(predicted_jodis),
            "Jodi_Act": "Pending",
            "Open_Match": "Pending",
            "Close_Match": "Pending",
            "Jodi_Match": "Pending",
            "Patti_Match": "Pending",
            "Model": row.get('Model', 'N/A')
        })
        continue

    actual_row = actual.iloc[0]
    actual_open = str(actual_row['Open']).strip()
    actual_close = str(actual_row['Close']).strip()
    actual_jodi = str(actual_row['Jodi']).strip().zfill(2)
    actual_patti = actual_open + actual_jodi[0] + actual_close

    open_match = actual_open in predicted_opens
    close_match = actual_close in predicted_closes
    jodi_match = actual_jodi in predicted_jodis
    patti_match = actual_patti in predicted_pattis

    matched.append({
        "Market": market,
        "Date": today,
        "Open_Pred": ','.join(predicted_opens),
        "Open_Act": actual_open,
        "Close_Pred": ','.join(predicted_closes),
        "Close_Act": actual_close,
        "Jodi_Pred": ','.join(predicted_jodis),
        "Jodi_Act": actual_jodi,
        "Open_Match": open_match,
        "Close_Match": close_match,
        "Jodi_Match": jodi_match,
        "Patti_Match": patti_match,
        "Model": row.get('Model', 'N/A')
    })

# Save to log
log_df = pd.DataFrame(matched)
log_df.to_csv(ACCURACY_LOG, mode='a', header=not os.path.exists(ACCURACY_LOG), index=False)

# Send Telegram summary
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
