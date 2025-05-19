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
messages = []

for _, row in pred_df.iterrows():
    market = row['Market']
    predicted_opens = [x.strip() for x in str(row.get('Open', '')).split(',')]
    predicted_closes = [x.strip() for x in str(row.get('Close', '')).split(',')]
    predicted_jodis = [x.strip().zfill(2) for x in str(row.get('Jodis', '')).split(',')]
    predicted_pattis = [x.strip() for x in str(row.get('Pattis', '')).split(',')]

    actual = today_actuals[today_actuals['Market'] == market]
    if actual.empty:
        continue  # No result yet

    actual_row = actual.iloc[0]
    actual_open = str(actual_row['Open']).strip()
    actual_close = str(actual_row['Close']).strip()
    actual_jodi = str(actual_row['Jodi']).strip().zfill(2)

    if actual_open == '' or actual_close == '' or actual_jodi == '':
        continue  # Incomplete result, skip

    actual_patti = actual_open + actual_jodi[0] + actual_close

    open_match = actual_open in predicted_opens
    close_match = actual_close in predicted_closes
    jodi_match = actual_jodi in predicted_jodis
    patti_match = actual_patti in predicted_pattis

    # Log the result
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

    # Send summary for this market
    messages.append(
        f"<b>{market}</b>\n"
        f"<b>Open:</b> {','.join(predicted_opens)} vs {actual_open} ({'✔' if open_match else '✘'})\n"
        f"<b>Close:</b> {','.join(predicted_closes)} vs {actual_close} ({'✔' if close_match else '✘'})\n"
        f"<b>Jodi:</b> {','.join(predicted_jodis)} vs {actual_jodi} ({'✔' if jodi_match else '✘'})\n"
        f"<b>Patti Match:</b> {'✔' if patti_match else '✘'}"
    )

# Save logs
if matched:
    log_df = pd.DataFrame(matched)
    log_df.to_csv(ACCURACY_LOG, mode='a', header=not os.path.exists(ACCURACY_LOG), index=False)

# Send message only if any result was found
if messages:
    final_message = "<b>Market Result Matched:</b>\n\n" + "\n\n".join(messages)
    send_telegram_message(final_message)
    print("Telegram message sent.")
else:
    print("No new market result available yet.")
