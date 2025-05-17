# satta_prediction_advanced.py

import os
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
from keras.preprocessing.sequence import TimeseriesGenerator
from collections import Counter
import joblib

TELEGRAM_BOT_TOKEN = "7121966371:AAEKHVrsqLRswXg64-6Nf3nid-Mbmlmmw5M"
CHAT_ID = "7621883960"
CSV_FILE = "satta_data.csv"
ACCURACY_LOG = "accuracy_log.csv"
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


def train_models(X, y):
    models = {}
    scores = {}

    rf = GridSearchCV(RandomForestClassifier(), {'n_estimators': [100, 200]})
    rf.fit(X, y)
    models['RandomForest'] = rf.best_estimator_
    scores['RandomForest'] = rf.score(X, y)

    xgb = GridSearchCV(XGBClassifier(), {'n_estimators': [100, 150]})
    xgb.fit(X, y)
    models['XGBoost'] = xgb.best_estimator_
    scores['XGBoost'] = xgb.score(X, y)

    return models[max(scores, key=scores.get)], scores


def predict_next(model, X_last):
    return model.predict(X_last)[0]


# Load data
try:
    df = pd.read_csv(CSV_FILE)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
except:
    df = pd.DataFrame(columns=['Date', 'Market', 'Open', 'Jodi', 'Close'])

# Scrape new data
for market, url in MARKETS.items():
    records = parse_table(url)
    for r in records:
        r['Market'] = market
    df = pd.concat([df, pd.DataFrame(records)], ignore_index=True)
    df.drop_duplicates(subset=['Date', 'Market'], inplace=True)

df.to_csv(CSV_FILE, index=False)

# Prediction date
today = datetime.today()
next_day = today + timedelta(days=1)
if next_day.weekday() == 6:  # Sunday
    next_day += timedelta(days=1)
predict_date = next_day.strftime("%d/%m/%Y")

# Prediction
final_predictions = []

for market in df['Market'].unique():
    mdf = df[df['Market'] == market].copy()
    mdf['Jodi'] = mdf['Jodi'].astype(str).str.zfill(2).astype(int)
    mdf = engineer_features(mdf)

    features = ['Prev_Jodi', 'Gap', 'Jodi_Pos1', 'Jodi_Pos2', 'Weekday']
    target = 'Jodi'

    model, scores = train_models(mdf[features], mdf[target])
    joblib.dump(model, f"model_{market}.pkl")

    X_last = mdf[features].tail(1)
    jodi_pred = predict_next(model, X_last)
    open_pred = str(jodi_pred).zfill(2)[0]
    close_pred = str(jodi_pred).zfill(2)[1]

    patti_candidates = [str(o)+str(jodi_pred)[0]+str(c) for o, c in zip(mdf['Open'].astype(str), mdf['Close'].astype(str)) if o.isdigit() and c.isdigit()]
    pattis = [p for p, _ in Counter(patti_candidates).most_common(4)]

    message = f"""
<b>{market.upper()}</b>
<b>{predict_date}</b>
<b>Open :</b> {open_pred}
<b>Close :</b> {close_pred}
<b>Jodi :</b> {str(jodi_pred).zfill(2)}
<b>Patti :</b> {', '.join(pattis)}
"""
    final_predictions.append(message)

send_telegram_message("\n\n".join(final_predictions))
print("Prediction sent.")
    
