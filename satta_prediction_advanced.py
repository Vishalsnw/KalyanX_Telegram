import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime
import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings("ignore")

TELEGRAM_TOKEN = "7121966371:AAEKHVrsqLRswXg64-6Nf3nid-Mbmlmmw5M"
CHAT_ID = "7621883960"

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }
    requests.post(url, data=data)

def load_data():
    df = pd.read_csv("satta_data.csv")
    df.dropna(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.sort_values("Date")
    return df

def preprocess_features(df, market):
    df_market = df[df["Market"] == market].copy()
    
    required_cols = ["Jodi", "Open", "Close"]
    for col in required_cols:
        if col not in df_market.columns or df_market[col].isnull().all():
            print(f"Skipping {market} — missing column: {col}")
            return None

    df_market = df_market.dropna(subset=required_cols)
    df_market = df_market.tail(60)

    df_market["Jodi"] = df_market["Jodi"].astype(str).str.zfill(2)
    df_market["OpenDigit"] = df_market["Jodi"].str[0].astype(int)
    df_market["CloseDigit"] = df_market["Jodi"].str[1].astype(int)
    df_market["Open"] = df_market["Open"].astype(int)
    df_market["Close"] = df_market["Close"].astype(int)
    df_market["Patti"] = df_market["Open"]  # You can switch to Close if preferred
    df_market["Weekday"] = df_market["Date"].dt.weekday
    return df_market

def build_lstm_model(input_shape, output_classes):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(output_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_models(df, target_col):
    X = df[["OpenDigit", "CloseDigit", "Weekday"]]
    y = df[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)

    xgb = XGBClassifier()
    xgb.fit(X_train, y_train)

    # LSTM
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    generator = TimeseriesGenerator(X_scaled, y, length=3, batch_size=1)
    lstm = build_lstm_model((3, X.shape[1]), len(np.unique(y)))
    lstm.fit(generator, epochs=10, verbose=0)

    return rf, xgb, lstm, scaler

def ensemble_predict(models, scaler, X_input):
    rf, xgb, lstm = models
    preds = []

    preds.append(int(rf.predict(X_input)[0]))
    preds.append(int(xgb.predict(X_input)[0]))

    X_scaled = scaler.transform(X_input)
    if len(X_scaled) < 3:
        X_scaled = np.vstack([X_scaled]*3)
    seq_input = np.array([X_scaled[-3:]])
    lstm_pred = lstm.predict(seq_input, verbose=0)
    preds.append(int(np.argmax(lstm_pred)))

    final = max(set(preds), key=preds.count)
    return final

def predict_for_market(df, market):
    df_market = preprocess_features(df, market)
    if df_market is None or len(df_market) < 10:
        print(f"Skipping {market} due to insufficient data.")
        return None

    latest_row = df_market.iloc[-1]
    features = [[latest_row["OpenDigit"], latest_row["CloseDigit"], latest_row["Weekday"]]]

    predictions = {}
    for col in ["Open", "Close", "Jodi", "Patti"]:
        try:
            rf, xgb, lstm, scaler = train_models(df_market, col)
            pred = ensemble_predict((rf, xgb, lstm), scaler, features)
            predictions[col] = pred
        except Exception as e:
            print(f"Error predicting {col} for {market}: {e}")
            predictions[col] = "?"

    return predictions

def main():
    df = load_data()
    markets = df["Market"].unique()
    today = datetime.now().strftime("%Y-%m-%d")
    prediction_log = []

    for market in markets:
        try:
            result = predict_for_market(df, market)
            if result:
                message = f"<b>{market} Prediction for {today}</b>\n"
                message += f"<b>Open:</b> {result['Open']}\n<b>Close:</b> {result['Close']}\n"
                message += f"<b>Jodi:</b> {result['Jodi']}\n<b>Patti:</b> {result['Patti']}"
                send_telegram_message(message)
                prediction_log.append([today, market, result['Open'], result['Close'], result['Jodi'], result['Patti'], "No"])
        except Exception as e:
            print(f"Error predicting for {market}: {e}")

    if prediction_log:
        df_log = pd.DataFrame(prediction_log, columns=["Date", "Market", "Open", "Close", "Jodi", "Patti", "Posted"])
        if os.path.exists("predictions.csv"):
            old = pd.read_csv("predictions.csv")
            df_log = pd.concat([old, df_log]).drop_duplicates(subset=["Date", "Market"], keep="last")
        df_log.to_csv("predictions.csv", index=False)

if __name__ == "__main__":
    main()
