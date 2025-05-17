import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import telegram
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

# --- Telegram Bot Setup ---
TELEGRAM_TOKEN = '7121966371:AAEKHVrsqLRswXg64-6Nf3nid-Mbmlmmw5M'
TELEGRAM_CHAT_ID = '7621883960'
bot = telegram.Bot(token=TELEGRAM_TOKEN)

# --- File paths ---
DATA_CSV = 'enhanced_satta_data.csv'   # Your main CSV file
SCALER_PATH = 'scaler.pkl'
MLP_MODEL_PATH = 'kalyan_mlp_model.pkl'  # Replace with your actual model name or path
KERAS_MODEL_PATH = 'kalyan_model.h5'      # Replace with your actual keras model path

# --- Markets to handle ---
MARKETS = ['kalyan', 'main-bazar', 'milan-day', 'milan-night', 'rajdhani-day', 'rajdhani-night', 'time-bazar']

# --- Date format in your CSV ---
DATE_FORMAT = '%d/%m/%Y'

def load_data():
    if not os.path.exists(DATA_CSV):
        logging.error(f'Data CSV file not found: {DATA_CSV}')
        return None
    df = pd.read_csv(DATA_CSV)
    # Convert date to datetime
    df['Date'] = pd.to_datetime(df['Date'], format=DATE_FORMAT)
    return df

def save_data(df):
    df.to_csv(DATA_CSV, index=False)
    logging.info('CSV file updated with new data.')

def feature_engineering(df):
    # Example: add your real features here, this is a placeholder
    df['open_sum'] = df['Open'] + df['Jodi']
    df['close_sum'] = df['Close'] + df['Jodi']
    # Add more features based on your earlier script...
    return df

def load_models():
    if not os.path.exists(SCALER_PATH) or not os.path.exists(MLP_MODEL_PATH) or not os.path.exists(KERAS_MODEL_PATH):
        logging.error('One or more model files missing.')
        return None, None, None
    scaler = joblib.load(SCALER_PATH)
    mlp_model = joblib.load(MLP_MODEL_PATH)
    keras_model = load_model(KERAS_MODEL_PATH)
    logging.info('Models loaded successfully.')
    return scaler, mlp_model, keras_model

def prepare_features(df, scaler):
    feature_cols = ['open_sum', 'close_sum']  # Replace with your real feature columns
    if len(df) == 0:
        raise ValueError('No data to prepare features.')
    X = df[feature_cols].values
    X_scaled = scaler.transform(X)
    return X_scaled

def predict_jodis(scaler, mlp_model, keras_model, df_market):
    # Prepare features
    X_scaled = prepare_features(df_market, scaler)

    # MLP predict probabilities (for top 10 jodis)
    probs = mlp_model.predict_proba(X_scaled)
    # Top 10 jodis by MLP model
    top_indices_mlp = np.argsort(probs[:,1])[-10:][::-1]

    # Keras model prediction (you might adjust this to your keras model's output)
    keras_preds = keras_model.predict(X_scaled).flatten()
    top_indices_keras = np.argsort(keras_preds)[-10:][::-1]

    # Combine or choose predictions (simple union here)
    combined_indices = np.unique(np.concatenate([top_indices_mlp, top_indices_keras]))

    top_jodis = df_market.iloc[combined_indices]['Jodi'].values.tolist()
    logging.info(f'Top predicted jodis: {top_jodis}')
    return top_jodis

def send_telegram_message(message):
    try:
        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
        logging.info('Telegram message sent.')
    except Exception as e:
        logging.error(f'Failed to send Telegram message: {e}')

def main():
    df = load_data()
    if df is None:
        return

    scaler, mlp_model, keras_model = load_models()
    if not scaler or not mlp_model or not keras_model:
        return

    # Feature engineering for full dataset
    df = feature_engineering(df)

    # Predict & notify for each market
    today = datetime.today().date()
    for market in MARKETS:
        df_market = df[df['Market'].str.lower() == market.lower()]
        if df_market.empty:
            logging.warning(f'No data for market: {market}')
            continue

        # Filter for the latest available date to predict next day
        last_date = df_market['Date'].max()
        if last_date >= today:
            logging.info(f'Market {market} is already updated for date {last_date.strftime(DATE_FORMAT)}')
            continue

        # Filter rows for last_date to prepare features
        df_last = df_market[df_market['Date'] == last_date]

        try:
            top_jodis = predict_jodis(scaler, mlp_model, keras_model, df_last)
        except Exception as e:
            logging.error(f'Prediction error for market {market}: {e}')
            continue

        # Compose Telegram message
        msg = f"Market: {market.title()}\nDate: {(last_date + timedelta(days=1)).strftime(DATE_FORMAT)}\nPredicted Top Jodis:\n"
        msg += ', '.join(str(j) for j in top_jodis)
        send_telegram_message(msg)

        # Append new prediction row with placeholder actual results for next date
        new_date = last_date + timedelta(days=1)
        new_row = {
            'Date': new_date.strftime(DATE_FORMAT),
            'Market': market,
            'Open': np.nan,    # Placeholder, actual results to be updated later
            'Jodi': top_jodis[0], # Top predicted jodi as example
            'Close': np.nan,
            'day_of_week': new_date.weekday(),
            'is_weekend': 1 if new_date.weekday() >= 5 else 0,
            'open_sum': np.nan,
            'close_sum': np.nan,
            'mirror_open': np.nan,
            'mirror_close': np.nan,
            'reverse_jodi': np.nan,
            'is_holiday': 0,
            'prev_jodi_distance': np.nan,
        }
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        save_data(df)

if __name__ == "__main__":
    main()
