# predict.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib, os
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import telegram

# Settings
CSV_FILE = "enhanced_satta_data.csv"
ACCURACY_LOG = "accuracy_log.csv"
MARKETS = ["Time Bazar", "Milan Day", "Rajdhani Day", "Kalyan", "Milan Night", "Rajdhani Night", "Main Bazar"]
TELEGRAM_TOKEN = "YOUR_TELEGRAM_TOKEN"
CHAT_ID = "YOUR_CHAT_ID"
TODAY = datetime.now().strftime("%d/%m/%Y")

def send_telegram_message(message):
    bot = telegram.Bot(token=TELEGRAM_TOKEN)
    bot.send_message(chat_id=CHAT_ID, text=message, parse_mode='HTML')

def preprocess_data(df, market):
    df = df[df['Market'] == market].dropna(subset=['Open', 'Close', 'Jodi', 'open_sum', 'mirror_open'])
    df['Open'] = df['Open'].astype(str).str.zfill(3)
    df['Close'] = df['Close'].astype(str).str.zfill(3)
    return df

def extract_features(df):
    features = ['open_sum', 'close_sum', 'mirror_open', 'mirror_close', 'prev_open_gap', 'prev_close_gap', 'is_weekend']
    return df[features], df['Jodi'].astype(str)

def train_and_predict(X, y, market):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # RandomForest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_score = accuracy_score(y_test, rf.predict(X_test))

    # XGBoost
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    xgb.fit(X_train, y_train)
    xgb_score = accuracy_score(y_test, xgb.predict(X_test))

    # LSTM (simplified)
    lstm_model = Sequential()
    lstm_model.add(LSTM(64, input_shape=(X_train.shape[1], 1)))
    lstm_model.add(Dense(100, activation='relu'))
    lstm_model.add(Dense(len(np.unique(y)), activation='softmax'))
    lstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    y_train_int = y_train.astype('category').cat.codes
    lstm_model.fit(np.expand_dims(X_train.values, axis=2), y_train_int, epochs=10, verbose=0)
    lstm_preds = lstm_model.predict(np.expand_dims(X_test.values, axis=2))
    lstm_preds_classes = np.argmax(lstm_preds, axis=1)
    lstm_score = accuracy_score(y_train.astype('category').cat.codes[:len(lstm_preds_classes)], lstm_preds_classes)

    best_model = max([(rf, rf_score), (xgb, xgb_score)], key=lambda x: x[1])[0]

    # Predict for tomorrow
    latest = X.tail(1)
    prediction = best_model.predict(latest)[0]
    return prediction, max(rf_score, xgb_score, lstm_score)

def update_accuracy_log(market, accuracy):
    today = datetime.now().strftime("%Y-%m-%d")
    new_entry = pd.DataFrame([[today, market, accuracy]], columns=["Date", "Market", "Accuracy"])
    if os.path.exists(ACCURACY_LOG):
        old = pd.read_csv(ACCURACY_LOG)
        new = pd.concat([old, new_entry], ignore_index=True)
    else:
        new = new_entry
    new.to_csv(ACCURACY_LOG, index=False)

def main():
    df = pd.read_csv(CSV_FILE)
    final_predictions = []

    for market in MARKETS:
        try:
            market_df = preprocess_data(df, market)
            if len(market_df) < 200: continue

            X, y = extract_features(market_df)
            prediction, accuracy = train_and_predict(X, y, market)
            update_accuracy_log(market, accuracy)
            final_predictions.append(f"<b>{market}</b>: Jodi <b>{prediction}</b> | Accuracy: {accuracy:.2%}")
        except Exception as e:
            print(f"Error in {market}: {e}")
    
    if final_predictions:
        full_message = "<b>Tomorrow's Satta Predictions</b>\n\n" + "\n".join(final_predictions)
    else:
        full_message = "No predictions generated due to data issues."
    
    send_telegram_message(full_message)

if __name__ == "__main__":
    main()
