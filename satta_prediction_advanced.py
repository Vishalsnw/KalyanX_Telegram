import pandas as pd
import numpy as np
import datetime
import telegram
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

TELEGRAM_TOKEN = "7121966371:AAEKHVrsqLRswXg64-6Nf3nid-Mbmlmmw5M"
CHAT_ID = "7621883960"

# Load data
df = pd.read_csv("enhanced_satta_data.csv")
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date'])

today = datetime.date.today()
ninety_days_ago = pd.to_datetime(today - datetime.timedelta(days=90))
df = df[df['Date'] >= ninety_days_ago]
df = df[df['Date'] < pd.to_datetime(today)]
df = df.sort_values(by=["Market", "Date"])

def prepare_features(df_market):
    df_market['DayOfWeek'] = df_market['Date'].dt.dayofweek
    df_market['Prev_Open'] = df_market['Open'].shift(1)
    df_market['Prev_Close'] = df_market['Close'].shift(1)
    df_market['Prev_Jodi'] = df_market['Jodi'].shift(1)
    df_market['Prev_Patti'] = df_market['Patti'].shift(1)
    df_market = df_market.dropna()
    features = ['DayOfWeek', 'Prev_Open', 'Prev_Close', 'Prev_Jodi', 'Prev_Patti']
    return df_market, features

def train_predict(df_market, features, target):
    X = df_market[features]
    y = df_market[target]
    if len(X) < 10:
        raise ValueError("Not enough data for training")
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict([X.iloc[-1].values])[0]
    return pred

def send_telegram_message(message):
    try:
        bot = telegram.Bot(token=TELEGRAM_TOKEN)
        bot.send_message(chat_id=CHAT_ID, text=message, parse_mode="HTML")
    except Exception as e:
        print(f"Telegram error: {e}")

def predict_all():
    markets = df['Market'].unique()
    full_msg = "<b>Tomorrow's Predictions:</b>\n\n"
    tomorrow = today + datetime.timedelta(days=1)

    for market in markets:
        try:
            df_market = df[df['Market'] == market].copy()
            df_market, features = prepare_features(df_market)

            open_pred = train_predict(df_market, features, 'Open')
            jodi_pred = train_predict(df_market, features, 'Jodi')
            close_pred = train_predict(df_market, features, 'Close')
            patti_pred = train_predict(df_market, features, 'Patti')

            full_msg += f"<b>{market}</b> ({tomorrow}):\n"
            full_msg += f"Open: <b>{open_pred}</b>\n"
            full_msg += f"Jodi: <b>{jodi_pred}</b>\n"
            full_msg += f"Close: <b>{close_pred}</b>\n"
            full_msg += f"Patti: <b>{patti_pred}</b>\n\n"
        except Exception as e:
            full_msg += f"<b>{market}</b>: Prediction failed ({str(e)})\n\n"

    send_telegram_message(full_msg)

if __name__ == "__main__":
    predict_all()
