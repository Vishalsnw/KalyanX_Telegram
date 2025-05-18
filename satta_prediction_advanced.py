# Imports
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime, timedelta
import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
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

# File Setup
def ensure_files():
    if not os.path.exists("enhanced_satta_data.csv"):
        df = pd.DataFrame(columns=[
            "Date", "Market", "Open", "Jodi", "Close", "day_of_week", "is_weekend",
            "open_sum", "close_sum", "mirror_open", "mirror_close",
            "reverse_jodi", "is_holiday", "prev_jodi_distance"
        ])
        df.to_csv("enhanced_satta_data.csv", index=False)

    if not os.path.exists("predictions.csv"):
        df = pd.DataFrame(columns=["Date", "Market", "Open", "Close", "Jodi", "Patti", "Posted"])
        df.to_csv("predictions.csv", index=False)

    if not os.path.exists("accuracy_log.csv"):
        df = pd.DataFrame(columns=["Date", "Market", "Open_Acc", "Close_Acc", "Jodi_Acc", "Patti_Acc"])
        df.to_csv("accuracy_log.csv", index=False)

# Load Dataset
def load_data():
    df = pd.read_csv("enhanced_satta_data.csv")
    df.dropna(subset=["Jodi", "Open", "Close"], inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.sort_values("Date")
    return df

# Feature Engineering
def add_features(df):
    df['day_of_week'] = df['Date'].dt.weekday
    df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
    df['open_sum'] = df['Open'].astype(str).str.zfill(3).apply(lambda x: sum(int(d) for d in x)%10)
    df['close_sum'] = df['Close'].astype(str).str.zfill(3).apply(lambda x: sum(int(d) for d in x)%10)
    df['mirror_open'] = df['Open'].astype(str).str.zfill(3).apply(lambda x: ''.join(str((10 - int(d)) % 10) for d in x))
    df['mirror_close'] = df['Close'].astype(str).str.zfill(3).apply(lambda x: ''.join(str((10 - int(d)) % 10) for d in x))
    df['reverse_jodi'] = df['Jodi'].astype(str).str.zfill(2).apply(lambda x: x[::-1])
    df['is_holiday'] = 0
    df['prev_jodi_distance'] = df['Jodi'].astype(str).str.zfill(2).astype(int).diff().fillna(0).abs()
    return df

# Append New Results
def append_actual_results_if_any():
    if not os.path.exists("predictions.csv"):
        return

    df = pd.read_csv("enhanced_satta_data.csv")
    df_pred = pd.read_csv("predictions.csv")
    df_pred["Date"] = pd.to_datetime(df_pred["Date"])
    latest_date = pd.to_datetime(df["Date"].max(), dayfirst=True) if not df.empty else pd.to_datetime("2000-01-01")

    new_results = df_pred[(df_pred["Posted"] == "Yes") & (df_pred["Date"] > latest_date)]

    if not new_results.empty:
        new_rows = []
        for _, row in new_results.iterrows():
            new_row = {
                "Date": row["Date"].strftime("%d/%m/%Y"),
                "Market": row["Market"],
                "Open": row["Open"],
                "Close": row["Close"],
                "Jodi": row["Jodi"],
            }
            new_rows.append(new_row)

        df_new = pd.DataFrame(new_rows)
        df_new['Date'] = pd.to_datetime(df_new['Date'], dayfirst=True)
        df_new = add_features(df_new)

        df = pd.concat([df, df_new])
        df = df.drop_duplicates(subset=["Date", "Market"], keep="last")
        df.to_csv("enhanced_satta_data.csv", index=False)

# Preprocess for model training
def preprocess_features(df, market):
    df_market = df[df["Market"] == market].copy()
    if df_market.empty: return None
    df_market["Jodi"] = df_market["Jodi"].astype(str).str.zfill(2)
    df_market["OpenDigit"] = df_market["Jodi"].str[0].astype(int)
    df_market["CloseDigit"] = df_market["Jodi"].str[1].astype(int)
    df_market["Patti"] = df_market["Open"]
    df_market["Weekday"] = df_market["Date"].dt.weekday
    return df_market.dropna()

# LSTM Model
def build_lstm_model(input_shape, output_classes):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(output_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Train Models
def train_models(df, target_col):
    X = df[["OpenDigit", "CloseDigit", "Weekday"]]
    if target_col == "Jodi":
        y_raw = df[target_col].astype(str).str.zfill(2)
    else:
        y_raw = df[target_col].astype(str)

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    if len(np.unique(y)) < 2:
        raise ValueError("Insufficient class diversity for training.")

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)

    xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    xgb.fit(X_train, y_train)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_train)
    X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
    lstm = build_lstm_model((1, X.shape[1]), len(np.unique(y)))
    lstm.fit(X_lstm, y_train, epochs=10, verbose=0)

    return rf, xgb, lstm, scaler, le

# Ensemble Prediction
def ensemble_predict(models, scaler, le, X_input):
    rf, xgb, lstm = models
    X_input = np.array(X_input).reshape(1, -1)
    preds = []

    preds.append(int(rf.predict(X_input)[0]))
    preds.append(int(xgb.predict(X_input)[0]))

    X_scaled = scaler.transform(X_input)
    X_lstm_input = X_scaled.reshape((1, 1, X_scaled.shape[1]))
    lstm_pred = lstm.predict(X_lstm_input, verbose=0)
    preds.append(int(np.argmax(lstm_pred)))

    final_encoded = max(set(preds), key=preds.count)
    return le.inverse_transform([final_encoded])[0]

# Predict for Market
def predict_for_market(df, market):
    df_market = preprocess_features(df, market)
    if df_market is None or len(df_market) < 20:
        print(f"Not enough data for market {market}")
        return None

    latest_row = df_market.iloc[-1]
    features = [latest_row["OpenDigit"], latest_row["CloseDigit"], latest_row["Weekday"]]
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

# Main Execution
def main():
    ensure_files()
    append_actual_results_if_any()
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

# Run
if __name__ == "__main__":
    main()
