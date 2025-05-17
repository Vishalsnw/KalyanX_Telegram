import pandas as pd
import numpy as np
import datetime
import logging
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import telegram

# Telegram bot setup
TELEGRAM_TOKEN = "7121966371:AAEKHVrsqLRswXg64-6Nf3nid-Mbmlmmw5M"
TELEGRAM_CHAT_ID = "7621883960"
bot = telegram.Bot(token=TELEGRAM_TOKEN)

def send_telegram_message(text):
    try:
        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=text)
    except Exception as e:
        logging.error(f"Telegram message send failed: {e}")

def load_data(file_path, market=None):
    df = pd.read_csv(file_path)
    if market:
        df = df[df['Market'] == market]
    return df

def prepare_features(df):
    # Example features: Use relevant columns from your CSV for prediction
    # Adjust these feature columns based on your actual CSV structure
    feature_cols = ['Open', 'Close', 'open_sum', 'close_sum', 'prev_jodi_distance']
    df = df.dropna(subset=feature_cols + ['Jodi'])
    X = df[feature_cols].values
    y = df['Jodi'].values
    return X, y

def train_and_save_models(X, y, market):
    logging.info(f"Training models for market {market}...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, f"scaler_{market}.pkl")

    mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
    mlp.fit(X_scaled, y)
    joblib.dump(mlp, f"{market}_mlp_model.pkl")

    num_classes = len(np.unique(y))
    model = Sequential([
        Dense(64, input_dim=X.shape[1], activation='relu'),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_scaled, y, epochs=20, batch_size=16, verbose=0)
    model.save(f"{market}_model.h5")
    logging.info(f"Models saved for market {market}")

def load_models(market):
    scaler = joblib.load(f"scaler_{market}.pkl")
    mlp = joblib.load(f"{market}_mlp_model.pkl")
    keras_model = load_model(f"{market}_model.h5")
    return scaler, mlp, keras_model

def predict_next_day(df, scaler, mlp, keras_model):
    # Use last row features to predict next day
    feature_cols = ['Open', 'Close', 'open_sum', 'close_sum', 'prev_jodi_distance']
    last_row = df.dropna(subset=feature_cols).iloc[-1]
    X_pred = last_row[feature_cols].values.reshape(1, -1)
    X_scaled = scaler.transform(X_pred)

    # sklearn MLP prediction
    proba = mlp.predict_proba(X_scaled)[0]
    top_indices = np.argsort(proba)[::-1][:10]
    predicted_jodis = mlp.classes_[top_indices]

    # keras model prediction (optional, can be combined or compared)
    keras_proba = keras_model.predict(X_scaled)
    keras_top_indices = keras_proba[0].argsort()[::-1][:10]
    keras_predicted_jodis = keras_top_indices  # these are class indices

    return predicted_jodis, proba[top_indices], keras_predicted_jodis

def main():
    logging.basicConfig(level=logging.INFO)

    data_file = "enhanced_satta_data.csv"
    markets = ['Kalyan', 'Main Bazar', 'Milan', 'Rajdhani', 'Time Bazar']

    for market in markets:
        logging.info(f"Processing market: {market}")
        df = load_data(data_file, market)
        if df.empty:
            logging.warning(f"No data for market {market}, skipping.")
            continue

        X, y = prepare_features(df)
        if len(X) == 0:
            logging.warning(f"No valid feature rows for market {market}, skipping.")
            continue

        import os
        scaler_file = f"scaler_{market}.pkl"
        mlp_file = f"{market}_mlp_model.pkl"
        keras_file = f"{market}_model.h5"

        if not (os.path.exists(scaler_file) and os.path.exists(mlp_file) and os.path.exists(keras_file)):
            logging.info(f"Model files missing for {market}, training new models.")
            train_and_save_models(X, y, market)

        scaler, mlp, keras_model = load_models(market)
        predicted_jodis, proba, keras_predicted_jodis = predict_next_day(df, scaler, mlp, keras_model)

        today_str = datetime.date.today().strftime("%d-%m-%Y")
        message = f"Market: {market}\nDate: {today_str}\nTop 10 Predicted Jodis:\n" + ", ".join(map(str, predicted_jodis))
        send_telegram_message(message)
        logging.info(f"Telegram message sent for {market}")

if __name__ == "__main__":
    main()
