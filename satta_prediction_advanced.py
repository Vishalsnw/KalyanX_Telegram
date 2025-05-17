import os
import pandas as pd
import numpy as np
import requests
import joblib
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")

# Constants
MARKETS = {
    "Kalyan": "https://dpbossattamatka.com/panel-chart-record/kalyan.php",
    "Main Bazar": "https://dpbossattamatka.com/panel-chart-record/main-bazar.php",
    "Time Bazar": "https://dpbossattamatka.com/panel-chart-record/time-bazar.php",
    "Milan Day": "https://dpbossattamatka.com/panel-chart-record/milan-day.php",
    "Rajdhani Day": "https://dpbossattamatka.com/panel-chart-record/rajdhani-day.php",
    "Milan Night": "https://dpbossattamatka.com/panel-chart-record/milan-night.php",
    "Rajdhani Night": "https://dpbossattamatka.com/panel-chart-record/rajdhani-night.php"
}
CSV_FILE = "satta_data.csv"
PRED_FILE = "today_predictions.csv"
BOT_TOKEN = "7121966371:AAEKHVrsqLRswXg64-6Nf3nid-Mbmlmmw5M"
CHAT_ID = "7621883960"

HEADERS = {"User-Agent": "Mozilla/5.0"}

def send_telegram_message(msg):
    try:
        requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                      data={"chat_id": CHAT_ID, "text": msg, "parse_mode": "HTML"})
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
                        date = (base_date + pd.Timedelta(days=i)).strftime("%d/%m/%Y")
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

def prepare_features(df):
    df['Open'] = df['Open'].astype(str).str.zfill(2)
    df['Close'] = df['Close'].astype(str).str.zfill(2)
    df['Jodi'] = df['Jodi'].astype(str).str.zfill(2)
    df['OpenDigit'] = df['Open'].str[0].astype(int)
    df['CloseDigit'] = df['Close'].str[-1].astype(int)
    df['JodiSum'] = df['Jodi'].astype(int) % 10
    df['Day'] = pd.to_datetime(df['Date'], dayfirst=True).dt.dayofweek
    return df

def train_predict(df, target_col):
    df = prepare_features(df)
    features = ['OpenDigit', 'CloseDigit', 'JodiSum', 'Day']
    df = df.dropna(subset=[target_col])
    X = df[features]
    y = df[target_col].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    models = {
        "RF": RandomForestClassifier(n_estimators=200),
        "XGB": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
        "LR": LogisticRegression()
    }

    best_model = None
    best_acc = 0
    best_name = ""

    for name, model in models.items():
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name

    best_model.fit(X, y)
    last_row = df.iloc[-1:][features]
    pred = best_model.predict(last_row)[0]
    return pred, best_name

def get_latest_data():
    all_data = []
    for market, url in MARKETS.items():
        records = parse_table(url)
        for r in records:
            r['Market'] = market
        all_data.extend(records)
    df = pd.DataFrame(all_data)
    df.to_csv(CSV_FILE, index=False)
    return df

def main():
    print("Running prediction script...")
    df = get_latest_data()
    today = datetime.today()
    predictions = []

    for market in MARKETS.keys():
        mdf = df[df['Market'] == market].sort_values('Date').dropna()
        if len(mdf) < 20: continue
        try:
            open_pred, open_model = train_predict(mdf, 'Open')
            close_pred, close_model = train_predict(mdf, 'Close')
            jodi_pred, jodi_model = train_predict(mdf, 'Jodi')
            patti = f"{open_pred}{jodi_pred[0]}{close_pred}"
            predictions.append({
                "Market": market,
                "Date": today.strftime("%d/%m/%Y"),
                "Open": open_pred,
                "Close": close_pred,
                "Jodi": jodi_pred,
                "Patti": patti,
                "Model": f"{open_model}/{close_model}/{jodi_model}"
            })
        except Exception as e:
            print(f"Error predicting for {market}: {e}")

    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(PRED_FILE, index=False)

    msg = "<b>Predictions for Tomorrow:</b>\n\n" + "\n\n".join(
        f"<b>{row['Market']}</b>\nOpen: <b>{row['Open']}</b>\nClose: <b>{row['Close']}</b>\nJodi: <b>{row['Jodi']}</b>\nPatti: <b>{row['Patti']}</b>"
        for _, row in pred_df.iterrows()
    )
    send_telegram_message(msg)

if __name__ == "__main__":
    main()
