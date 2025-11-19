import pandas as pd
import numpy as np
import telegram
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import warnings
import os

# === CONFIG ===
warnings.filterwarnings("ignore")
TELEGRAM_TOKEN = "8050429062:AAFPLG9NuPnkDjVZyLUeg35Tlg4ArKisLbQ"
CHAT_ID = "-1002573892631"
MARKETS = ["Time Bazar", "Milan Day", "Rajdhani Day", "Kalyan", "Milan Night", "Rajdhani Night", "Main Bazar"]
DATA_FILE = "satta_data.csv"
PRED_FILE = "today_ml_prediction.csv"
ACCURACY_FILE = "prediction_accuracy.csv"

# === TELEGRAM ===
def send_telegram_message(message):
    try:
        bot = telegram.Bot(token=TELEGRAM_TOKEN)
        bot.send_message(chat_id=CHAT_ID, text=message, parse_mode=telegram.ParseMode.HTML)
    except Exception as e:
        print("Telegram Error:", e)

# === UTILS ===
def patti_to_digit(patti):
    return sum(int(d) for d in str(int(patti)).zfill(3)) % 10

def generate_pattis(open_vals, close_vals):
    pattis = set()
    for val in open_vals + close_vals:
        try:
            base = int(val)
            digits = list(str(base).zfill(3))
            sorted_digits = ''.join(sorted(digits))
            pattis.add(sorted_digits)
        except:
            continue
    return sorted(pattis)[:4]

def next_prediction_date():
    today = datetime.now()
    tomorrow = today + timedelta(days=1)
    if tomorrow.weekday() == 6:  # Sunday
        return (tomorrow + timedelta(days=1)).strftime("%d/%m/%Y")
    return tomorrow.strftime("%d/%m/%Y")

# === MARKOV CHAIN MODEL ===
class MarkovChain:
    def __init__(self, order=2):
        self.order = order
        self.transitions = defaultdict(Counter)
        self.states = set()
    
    def fit(self, sequence):
        """Train Markov chain on a sequence of states"""
        if len(sequence) <= self.order:
            return
        
        for i in range(len(sequence) - self.order):
            state = tuple(sequence[i:i + self.order])
            next_state = sequence[i + self.order]
            self.transitions[state][next_state] += 1
            self.states.add(state)
    
    def predict_next(self, current_state, top_k=2):
        """Predict next states given current state"""
        if len(current_state) != self.order:
            current_state = current_state[-self.order:]
        
        current_state = tuple(current_state)
        
        if current_state not in self.transitions:
            # If state not found, try to find similar states or return random
            return self._fallback_prediction(top_k)
        
        next_states = self.transitions[current_state]
        total = sum(next_states.values())
        
        # Get top k most likely next states
        predictions = []
        for state, count in next_states.most_common(top_k * 2):  # Get extra for filtering
            probability = count / total
            predictions.append((state, probability))
        
        return predictions[:top_k]
    
    def _fallback_prediction(self, top_k):
        """Fallback when current state is not in training data"""
        # Return most common states from entire dataset
        all_states = []
        for state_counter in self.transitions.values():
            for state, count in state_counter.items():
                all_states.append((state, count))
        
        if not all_states:
            return [(np.random.randint(0, 10), 0.1) for _ in range(top_k)]
        
        counter = Counter()
        for state, count in all_states:
            counter[state] += count
        
        return [(state, count/sum(counter.values())) for state, count in counter.most_common(top_k)]

# === LOAD DATA ===
def load_data():
    df = pd.read_csv(DATA_FILE)
    df["Market"] = df["Market"].astype(str).str.strip()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["Date", "Market", "Open", "Close", "Jodi"])
    df["Open"] = pd.to_numeric(df["Open"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df["Jodi"] = df["Jodi"].astype(str).str.zfill(2).str[-2:]
    return df.dropna()

# === FEATURE ENGINEERING FOR MARKOV CHAIN ===
def prepare_markov_features(df):
    df = df.sort_values("Date").copy()
    
    # Convert numbers to single digits for Markov chain
    df["Open_Digit"] = df["Open"].apply(patti_to_digit)
    df["Close_Digit"] = df["Close"].apply(patti_to_digit)
    df["Jodi_Digit1"] = df["Jodi"].str[0].astype(int)
    df["Jodi_Digit2"] = df["Jodi"].str[1].astype(int)
    
    return df

# === MARKOV CHAIN TRAINING AND PREDICTION ===
def train_markov_models(df_market):
    """Train separate Markov chains for Open, Close, and Jodi digits"""
    if len(df_market) < 5:
        return None, None, None, "Insufficient data"
    
    df_market = prepare_markov_features(df_market)
    
    # Train Markov chain for Open digits
    open_chain = MarkovChain(order=2)
    open_sequence = df_market["Open_Digit"].tolist()
    open_chain.fit(open_sequence)
    
    # Train Markov chain for Close digits
    close_chain = MarkovChain(order=2)
    close_sequence = df_market["Close_Digit"].tolist()
    close_chain.fit(close_sequence)
    
    # Train Markov chain for Jodi digits
    jodi_chain1 = MarkovChain(order=2)
    jodi_chain2 = MarkovChain(order=2)
    jodi_sequence1 = df_market["Jodi_Digit1"].tolist()
    jodi_sequence2 = df_market["Jodi_Digit2"].tolist()
    jodi_chain1.fit(jodi_sequence1)
    jodi_chain2.fit(jodi_sequence2)
    
    return open_chain, close_chain, jodi_chain1, jodi_chain2

def predict_with_markov(open_chain, close_chain, jodi_chain1, jodi_chain2, df_market):
    """Make predictions using trained Markov chains"""
    if not all([open_chain, close_chain, jodi_chain1, jodi_chain2]):
        return None, None, None, "Model training failed"
    
    df_market = prepare_markov_features(df_market)
    
    # Get recent history for prediction
    recent_data = df_market.tail(3)
    
    # Predict Open digits
    open_history = recent_data["Open_Digit"].tolist()
    open_preds = open_chain.predict_next(open_history, top_k=2)
    open_vals = [pred[0] for pred in open_preds]
    
    # Predict Close digits
    close_history = recent_data["Close_Digit"].tolist()
    close_preds = close_chain.predict_next(close_history, top_k=2)
    close_vals = [pred[0] for pred in close_preds]
    
    # Predict Jodi digits and combine
    jodi_history1 = recent_data["Jodi_Digit1"].tolist()
    jodi_history2 = recent_data["Jodi_Digit2"].tolist()
    
    jodi_preds1 = jodi_chain1.predict_next(jodi_history1, top_k=5)
    jodi_preds2 = jodi_chain2.predict_next(jodi_history2, top_k=5)
    
    # Generate top jodis by combining digit predictions
    jodi_vals = []
    for digit1, prob1 in jodi_preds1:
        for digit2, prob2 in jodi_preds2:
            jodi = f"{digit1}{digit2}"
            jodi_vals.append((jodi, prob1 * prob2))
    
    # Sort by probability and take top 10
    jodi_vals.sort(key=lambda x: x[1], reverse=True)
    top_jodis = [jodi[0] for jodi in jodi_vals[:10]]
    
    # Convert digit predictions back to 3-digit format (using most common patterns)
    open_3digit = [find_common_3digit(df_market, digit, 'Open') for digit in open_vals]
    close_3digit = [find_common_3digit(df_market, digit, 'Close') for digit in close_vals]
    
    return open_3digit, close_3digit, top_jodis, "Prediction successful"

def find_common_3digit(df, target_digit, column):
    """Find most common 3-digit number that reduces to target digit"""
    if column == 'Open':
        candidates = df[df["Open_Digit"] == target_digit]["Open"].value_counts()
    else:  # Close
        candidates = df[df["Close_Digit"] == target_digit]["Close"].value_counts()
    
    if not candidates.empty:
        return int(candidates.index[0])
    else:
        # Fallback: generate a random 3-digit number with correct digit sum
        return generate_3digit_with_digit(target_digit)

def generate_3digit_with_digit(target_digit):
    """Generate a 3-digit number that reduces to target digit"""
    # Simple approach: use pattern like 100 + target_digit * 11
    base = 100 + target_digit * 11
    return base if base <= 999 else 100 + target_digit * 10

# === TRAIN + PREDICT ===
def train_and_predict(df, market, prediction_date):
    df_market = df[df["Market"] == market].copy()
    if len(df_market) < 6:
        return None, None, None, "Insufficient data"
    
    # Train Markov chain models
    open_chain, close_chain, jodi_chain1, jodi_chain2 = train_markov_models(df_market)
    
    # Make predictions
    open_vals, close_vals, jodis, status = predict_with_markov(
        open_chain, close_chain, jodi_chain1, jodi_chain2, df_market
    )
    
    return open_vals, close_vals, jodis, status

# === MAIN ===
def main():
    df = load_data()
    prediction_date = next_prediction_date()
    full_msg = f"<b>📅 Satta Predictions — {prediction_date}</b>\n"
    full_msg += "<i>Using Markov Chain Model</i>\n"

    try:
        df_existing = pd.read_csv(PRED_FILE)
    except FileNotFoundError:
        df_existing = pd.DataFrame()

    try:
        df_acc = pd.read_csv(ACCURACY_FILE)
    except FileNotFoundError:
        df_acc = pd.DataFrame()

    new_preds = []

    for market in MARKETS:
        open_vals, close_vals, jodis, status = train_and_predict(df, market, prediction_date)

        if not open_vals or not close_vals or not jodis:
            full_msg += f"\n🔸 <b>{market}</b>\n<i>⚠️ {status}</i>\n"
            continue

        open_digits = [str(patti_to_digit(val)) for val in open_vals]
        close_digits = [str(patti_to_digit(val)) for val in close_vals]
        pattis = generate_pattis(open_vals, close_vals)

        # Card-like section
        full_msg += (
            f"\n🔸 <b>{market}</b>\n"
            f"🔹 <b>Open:</b> <code>{', '.join(open_digits)}</code>\n"
            f"🔹 <b>Close:</b> <code>{', '.join(close_digits)}</code>\n"
            f"🎰 <b>Pattis:</b> <code>{', '.join(pattis)}</code>\n"
            f"🔟 <b>Top Jodis:</b> <code>{', '.join(jodis)}</code>\n"
        )

        new_preds.append({
            "Market": market,
            "Date": prediction_date,
            "Open": ", ".join(open_digits),
            "Close": ", ".join(close_digits),
            "Pattis": ", ".join(pattis),
            "Jodis": ", ".join(jodis)
        })

        df_acc = pd.concat([df_acc, pd.DataFrame([{
            "Date": prediction_date,
            "Market": market,
            "Pred_Open": open_vals,
            "Pred_Close": close_vals,
            "Pred_Jodis": jodis
        }])], ignore_index=True)

    # Save predictions
    for row in new_preds:
        df_existing = df_existing[~(
            (df_existing['Market'] == row['Market']) &
            (df_existing['Date'] == row['Date'])
        )]
    df_combined = pd.concat([df_existing, pd.DataFrame(new_preds)], ignore_index=True)
    df_combined.to_csv(PRED_FILE, index=False)
    df_acc.to_csv(ACCURACY_FILE, index=False)

    # Send to Telegram
    send_telegram_message(full_msg)

if __name__ == "__main__":
    main()
