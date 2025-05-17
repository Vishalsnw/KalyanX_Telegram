# satta_prediction_advanced.py (Smart & Self-Learning)

import os
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from keras.models import Sequential
from keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler
from collections import Counter
import joblib
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN"
CHAT_ID = "YOUR_CHAT_ID"
CSV_FILE = "satta_data.csv"
PRED_FILE = "today_predictions.csv"
ACCURACY_LOG = "accuracy_log.csv"
MODEL_TRACK = "model_performance.json"
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

if not os.path.exists(MODEL_TRACK):
    with open(MODEL_TRACK, 'w') as f:
        json.dump({}, f)

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

def engineer_features(df):
    df['Weekday'] = df['Date'].dt.weekday
    df['Prev_Jodi'] = df['Jodi'].shift(1).fillna(0).astype(int)
    df['Gap'] = (df['Jodi'] - df['Prev_Jodi']).abs()
    df['Jodi_Pos1'] = df['Jodi'].astype(str).str.zfill(2).str[0].astype(int)
    df['Jodi_Pos2'] = df['Jodi'].astype(str).str.zfill(2).str[1].astype(int)
    return df.dropna()

def train_lstm(X, y):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    y = np.array(y)
    generator = TimeseriesGenerator(X_scaled, y, length=10, batch_size=1)
    if len(generator) == 0:
        return None, 0
    model = Sequential([
        LSTM(32, activation='relu', input_shape=(10, X.shape[1])),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(generator, epochs=20, verbose=0)
    pred = model.predict(X_scaled[-10:].reshape(1, 10, X.shape[1]))[0][0]
    pred_rounded = int(round(pred))
    return model, pred_rounded

def train_models(X, y, market):
    models = {}
    scores = {}

    rf = GridSearchCV(RandomForestClassifier(), {'n_estimators': [100, 150]})
    rf.fit(X, y)
    rf_pred = rf.predict(X.tail(1))[0]
    models['RandomForest'] = rf
    scores['RandomForest'] = rf.score(X, y)

    xgb = GridSearchCV(XGBClassifier(), {'n_estimators': [100, 150]}, verbosity=0)
    xgb.fit(X, y)
    xgb_pred = xgb.predict(X.tail(1))[0]
    models['XGBoost'] = xgb
    scores['XGBoost'] = xgb.score(X, y)

    lstm_model, lstm_pred = train_lstm(X, y)
    if lstm_model:
        models['LSTM'] = lstm_model
        scores['LSTM'] = accuracy_score([y.iloc[-1]], [lstm_pred])

    best_model = max(scores, key=scores.get)
    with open(MODEL_TRACK, 'r+') as f:
        perf = json.load(f)
        perf[market] = best_model
        f.seek(0)
        json.dump(perf, f, indent=2)
        f.truncate()

    print(f"{market}: Best model - {best_model} ({scores[best_model]:.2f})")
    return models[best_model], best_model, locals().get(f"{best_model.lower()}_pred")

# --- Main Flow ---
try:
    df = pd.read_csv(CSV_FILE)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
except:
    df = pd.DataFrame(columns=['Date', 'Market', 'Open', 'Jodi', 'Close'])

for market, url in MARKETS.items():
    records = parse_table(url)
    for r in records:
        r['Market'] = market
    df = pd.concat([df, pd.DataFrame(records)], ignore_index=True)
    df.drop_duplicates(subset=['Date', 'Market'], inplace=True)

df.to_csv(CSV_FILE, index=False)

today = datetime.today()
next_day = today + timedelta(days=1)
if next_day.weekday() == 6:
    next_day += timedelta(days=1)
predict_date = next_day.strftime("%d/%m/%Y")

predictions = []
for market in df['Market'].unique():
    mdf = df[df['Market'] == market].copy()
    mdf['Jodi'] = mdf['Jodi'].astype(str).str.zfill(2).astype(int)
    mdf['Date'] = pd.to_datetime(mdf['Date'], dayfirst=True)
    mdf = engineer_features(mdf)

    if len(mdf) < 20: continue
    features = ['Prev_Jodi', 'Gap', 'Jodi_Pos1', 'Jodi_Pos2', 'Weekday']
    X = mdf[features]
    y = mdf['Jodi']

    model, model_type, jodi_pred = train_models(X, y, market)

    open_pred = str(jodi_pred).zfill(2)[0]
    close_pred = str(jodi_pred).zfill(2)[1]

    pattis = [str(o)+str(jodi_pred)[0]+str(c) for o, c in zip(mdf['Open'].astype(str), mdf['Close'].astype(str)) if o.isdigit() and c.isdigit()]
    pattis = [p for p, _ in Counter(pattis).most_common(3)]

    predictions.append({
        "Market": market,
        "Date": predict_date,
        "Open": open_pred,
        "Close": close_pred,
        "Jodi": str(jodi_pred).zfill(2),
        "Patti": pattis,
        "Model": model_type
    })

message = "\n\n".join(
    f"<b>{p['Market']}</b>\n<b>{p['Date']}</b>\n<b>Model:</b> {p['Model']}\n<b>Open:</b> {p['Open']}\n<b>Close:</b> {p['Close']}\n<b>Jodi:</b> {p['Jodi']}\n<b>Patti:</b> {', '.join(p['Patti'])}"
    for p in predictions
)

send_telegram_message(message)

pd.DataFrame(predictions).to_csv(PRED_FILE, index=False)
print("Predictions sent and logged.")
                        
