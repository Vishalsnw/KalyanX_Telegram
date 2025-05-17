import os
import pandas as pd
import numpy as np
import datetime
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
import pickle
from telegram import Bot
import requests
from bs4 import BeautifulSoup

# --- CONFIG ---
MARKET = "Kalyan"
CSV_FILE = "enhanced_satta_data.csv"
PRED_FILE = "today_prediction.csv"
TELEGRAM_TOKEN = "7121966371:AAEKHVrsqLRswXg64-6Nf3nid-Mbmlmmw5M"
CHAT_ID = "7621883960"

# --- STEP 1: SCRAPE LATEST RESULTS ---
def scrape_kalyan_result():
    url = "https://dpbossattamatka.com/panel-chart-record/kalyan.php"
    res = requests.get(url)
    soup = BeautifulSoup(res.text, "html.parser")
    rows = soup.select("table tr")[1:]
    data = []
    for row in rows:
        cols = row.find_all("td")
        if len(cols) >= 4:
            date = cols[0].text.strip()
            open_patti = cols[1].text.strip()
            jodi = cols[2].text.strip()
            close_patti = cols[3].text.strip()
            data.append([date, open_patti, jodi, close_patti])
    return data

# --- STEP 2: UPDATE CSV ---
def update_csv_with_scraped(data):
    df = pd.read_csv(CSV_FILE)
    existing_dates = df[df["Market"] == MARKET]["Date"].unique()

    new_rows = []
    for date, open_patti, jodi, close_patti in data:
        if date not in existing_dates:
            row = {
                "Date": date,
                "Market": MARKET,
                "Open": open_patti,
                "Jodi": jodi,
                "Close": close_patti
            }
            new_rows.append(row)

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        df = pd.concat([df, new_df], ignore_index=True)
        df.to_csv(CSV_FILE, index=False)
        print(f"[INFO] Appended {len(new_rows)} new rows.")
    else:
        print("[INFO] No new rows to add.")

# --- STEP 3: FEATURE ENGINEERING ---
def create_features(df):
    df["day_of_week"] = pd.to_datetime(df["Date"], dayfirst=True).dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    df["open_sum"] = df["Open"].astype(str).str[:1].astype(int) + df["Open"].astype(str).str[1:2].astype(int)
    df["close_sum"] = df["Close"].astype(str).str[:1].astype(int) + df["Close"].astype(str).str[1:2].astype(int)
    df["mirror_open"] = df["Open"].astype(str).apply(lambda x: str(9 - int(x[0])) + str(9 - int(x[1])) if len(x) == 2 else "00")
    df["mirror_close"] = df["Close"].astype(str).apply(lambda x: str(9 - int(x[0])) + str(9 - int(x[1])) if len(x) == 2 else "00")
    df["reverse_jodi"] = df["Jodi"].astype(str).apply(lambda x: x[::-1])
    df["is_holiday"] = 0
    df["prev_jodi_distance"] = df["Jodi"].astype(int).diff().abs().fillna(0)
    return df

# --- STEP 4: TRAIN + SAVE MODELS ---
def train_models(df):
    df = df[df["Market"] == MARKET].dropna()
    df = create_features(df)
    feature_cols = ["day_of_week", "is_weekend", "open_sum", "close_sum", "prev_jodi_distance"]
    X = df[feature_cols]
    y = df["Jodi"].astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000)
    mlp.fit(X_scaled, y)

    keras_model = Sequential([
        Dense(64, activation='relu', input_shape=(X_scaled.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    keras_model.compile(optimizer='adam', loss='mse')
    keras_model.fit(X_scaled, y, epochs=50, verbose=0)

    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open("kalyan_mlp_model.pkl", "wb") as f:
        pickle.dump(mlp, f)
    keras_model.save("kalyan_model.h5")

    return scaler, mlp, keras_model

# --- STEP 5: PREDICT NEXT JODI ---
def predict_next(df, scaler, mlp, keras_model):
    df = df[df["Market"] == MARKET]
    last_row = df.sort_values("Date").iloc[-1:]
    last_row = create_features(last_row)

    X_next = last_row[["day_of_week", "is_weekend", "open_sum", "close_sum", "prev_jodi_distance"]]
    X_scaled = scaler.transform(X_next)

    proba = mlp.predict_proba(X_scaled)[0]
    top_10 = np.argsort(proba)[-10:][::-1]

    keras_pred = int(round(keras_model.predict(X_scaled)[0][0]))

    preds = {
        "mlp_top_10": [str(j).zfill(2) for j in top_10],
        "keras_pred": str(keras_pred).zfill(2)
    }

    today = datetime.datetime.now().strftime("%d/%m/%Y")
    pd.DataFrame([{
        "Date": today,
        "Market": MARKET,
        "Top10_MLP": ",".join(preds["mlp_top_10"]),
        "Keras": preds["keras_pred"]
    }]).to_csv(PRED_FILE, index=False)

    return preds

# --- STEP 6: TELEGRAM ALERT ---
def send_telegram(preds):
    message = f"**KALYAN Prediction**\n\nTop 10 Jodis (MLP): {', '.join(preds['mlp_top_10'])}\nKeras Jodi: {preds['keras_pred']}"
    Bot(token=TELEGRAM_TOKEN).send_message(chat_id=CHAT_ID, text=message, parse_mode='Markdown')

# --- RUN ALL ---
if __name__ == "__main__":
    scraped_data = scrape_kalyan_result()
    update_csv_with_scraped(scraped_data)
    df = pd.read_csv(CSV_FILE)
    scaler, mlp, keras_model = train_models(df)
    preds = predict_next(df, scaler, mlp, keras_model)
    send_telegram(preds)
    print("[DONE] Prediction complete and Telegram sent.")
