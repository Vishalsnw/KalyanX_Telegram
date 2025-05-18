# Imports
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
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# Telegram Setup
TELEGRAM_TOKEN = "7121966371:AAEKHVrsqLRswXg64-6Nf3nid-Mbmlmmw5M"
CHAT_ID = "7621883960"

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {"chat_id": CHAT_ID, "text": message, "parse_mode": "HTML"}
    try:
        requests.post(url, data=data)
    except Exception as e:
        print(f"Telegram error: {e}")

# Data Load
def load_data():
    df = pd.read_csv("satta_data.csv")
    df.dropna(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.sort_values("Date")
    return df

# Feature Engineering
def preprocess_features(df, market):
    df_market = df[df["Market"] == market].copy()
    df_market = df_market.dropna(subset=["Jodi", "Open", "Close"])
    df_market = df_market.tail(60)

    if len(df_market) < 10:
        return None

    df_market["Jodi"] = df_market["Jodi"].astype(str).str.zfill(2)
    df_market["OpenDigit"] = df_market["Jodi"].str[0].astype(int)
    df_market["CloseDigit"] = df_market["Jodi"].str[1].astype(int)
    df_market["Open"] = df_market["Open"].astype(int)
    df_market["Close"] = df_market["Close"].astype(int)
    df_market["Patti"] = df_market["Open"]
    df_market["Weekday"] = df_market["Date"].dt.weekday
    df_market["PrevJodi"] = df_market["Jodi"].shift(1).fillna("00").astype(str).str.zfill(2)

    return df_market.dropna()

# Build LSTM Model
def build_lstm_model(input_shape, output_classes):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(32))
    model.add(Dropout(0.2))
    model.add(Dense(output_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Train All 3 Models
def train_models(df, target_col):
    X = df[["OpenDigit", "CloseDigit", "Weekday"]]
    y_raw = df[target_col].astype(str)

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)

    xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    xgb.fit(X_train, y_train)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    if len(X_scaled) < 4:
        raise Exception("Not enough data for LSTM.")

    generator = TimeseriesGenerator(X_scaled, y, length=3, batch_size=1)
    lstm = build_lstm_model((3, X.shape[1]), len(np.unique(y)))
    lstm.fit(generator, epochs=10, verbose=0)

    return rf, xgb, lstm, scaler, le

# Ensemble Logic
def ensemble_predict(models, scaler, le, X_input):
    rf, xgb, lstm = models
    preds = [int(rf.predict(X_input)[0]), int(xgb.predict(X_input)[0])]

    X_scaled = scaler.transform(X_input)
    if len(X_scaled) < 3:
        X_scaled = np.vstack([X_scaled] * 3)
    seq_input = np.array([X_scaled[-3:]])
    lstm_pred = lstm.predict(seq_input, verbose=0)
    preds.append(int(np.argmax(lstm_pred)))

    final_encoded = max(set(preds), key=preds.count)
    return le.inverse_transform([final_encoded])[0]

# Prediction for a Single Market
def predict_for_market(df, market):
    df_market = preprocess_features(df, market)
    if df_market is None or len(df_market) < 10:
        return None

    latest_row = df_market.iloc[-1]
    features = [[latest_row["OpenDigit"], latest_row["CloseDigit"], latest_row["Weekday"]]]

    predictions = {}
    for col in ["Open", "Close", "Jodi", "Patti"]:
        try:
            rf, xgb, lstm, scaler, le = train_models(df_market, col)
            pred = ensemble_predict((rf, xgb, lstm), scaler, le, features)
            predictions[col] = pred
        except Exception as e:
            print(f"{market} - {col} error: {e}")
            predictions[col] = "?"
    return predictions

# Accuracy Logging
def log_accuracy(df_actual, df_pred):
    df_actual["Date"] = pd.to_datetime(df_actual["Date"])
    df_pred["Date"] = pd.to_datetime(df_pred["Date"])

    merged = pd.merge(df_actual, df_pred, on=["Date", "Market"], suffixes=('_actual', '_pred'))
    acc_logs = []
    for _, row in merged.iterrows():
        acc_logs.append([
            row["Date"].strftime("%Y-%m-%d"),
            row["Market"],
            int(row["Open_actual"] == row["Open_pred"]),
            int(row["Close_actual"] == row["Close_pred"]),
            int(row["Jodi_actual"] == row["Jodi_pred"]),
            int(row["Patti_actual"] == row["Patti_pred"])
        ])
    df_log = pd.DataFrame(acc_logs, columns=["Date", "Market", "Open_Acc", "Close_Acc", "Jodi_Acc", "Patti_Acc"])
    if os.path.exists("accuracy_log.csv"):
        old = pd.read_csv("accuracy_log.csv")
        df_log = pd.concat([old, df_log]).drop_duplicates(subset=["Date", "Market"], keep="last")
    df_log.to_csv("accuracy_log.csv", index=False)

# Main Function
def main():
    df = load_data()
    today = datetime.now().strftime("%Y-%m-%d")
    markets = df["Market"].unique()
    prediction_log = []

    for market in markets:
        try:
            result = predict_for_market(df, market)
            if result:
                message = (
                    f"<b>{market} Prediction for {today}</b>\n"
                    f"<b>Open:</b> {result['Open']}\n"
                    f"<b>Close:</b> {result['Close']}\n"
                    f"<b>Jodi:</b> {result['Jodi']}\n"
                    f"<b>Patti:</b> {result['Patti']}"
                )
                send_telegram_message(message)
                prediction_log.append([
                    today, market, result["Open"], result["Close"], result["Jodi"], result["Patti"], "No"
                ])
        except Exception as e:
            print(f"Failed prediction for {market}: {e}")

    if prediction_log:
        df_pred = pd.DataFrame(prediction_log, columns=["Date", "Market", "Open", "Close", "Jodi", "Patti", "Posted"])
        if os.path.exists("predictions.csv"):
            old = pd.read_csv("predictions.csv")
            df_pred = pd.concat([old, df_pred]).drop_duplicates(subset=["Date", "Market"], keep="last")
        df_pred.to_csv("predictions.csv", index=False)

if __name__ == "__main__":
    main()
