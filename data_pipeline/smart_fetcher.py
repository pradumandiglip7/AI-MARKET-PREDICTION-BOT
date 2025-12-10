import os
import time
import requests
import pandas as pd
import ta
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    print("❌ Error: API_KEY not found in .env")
    exit()

BASE_URL = "https://api.twelvedata.com"

# REMOVED 'SPX' because it is a paid-only symbol
TICKERS = [
    "EUR/USD", "GBP/USD", "USD/ZAR", "EUR/INR", "EUR/GBP",
    "XAU/USD", "XAG/USD",
    "BTC/USD", "ETH/USD", "SHIB/USD", "DOGE/USD", "SOL/USD", "LTC/USD", "XRP/USD",
    "NVDA", "AAPL", "GOOG", "INTC", "NIO", "AMZN"
]

def fetch_ohlc_history(symbol):
    """Fetch the last 50 hours of data."""
    url = f"{BASE_URL}/time_series"
    params = {
        "symbol": symbol,
        "interval": "1h",
        "outputsize": 50,
        "apikey": API_KEY
    }
    
    try:
        resp = requests.get(url, params=params)
        data = resp.json()
        
        if "values" not in data:
            print(f"⚠️ Error fetching {symbol}: {data.get('message', 'Unknown Error')}")
            return None
            
        df = pd.DataFrame(data["values"])
        
        # --- FIX 1: Handle Missing Volume ---
        # Some assets (Forex/Crypto) might not return volume. We fill it with 0.
        if 'volume' not in df.columns:
            df['volume'] = 0

        # Convert text numbers to real numbers
        cols = ["open", "high", "low", "close", "volume"]
        df[cols] = df[cols].astype(float)
        
        # Flip data: Oldest first, Newest last
        df = df.iloc[::-1].reset_index(drop=True)
        return df

    except Exception as e:
        print(f"❌ Connection error for {symbol}: {e}")
        return None

def main():
    print("⬇️ Starting Smart Data Fetch (Robust Mode)...")
    results = []

    for symbol in TICKERS:
        print(f"⏳ Fetching {symbol}...")
        
        # 1. Get Raw Data
        df = fetch_ohlc_history(symbol)
        
        if df is not None and not df.empty:
            # 2. CALCULATE INDICATORS LOCALLY
            try:
                # RSI (14)
                df['rsi'] = ta.momentum.rsi(df['close'], window=14)
                
                # MACD (12, 26, 9)
                macd_obj = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
                df['macd'] = macd_obj.macd()
                df['macd_signal'] = macd_obj.macd_signal()
                
                # --- FIX 2: Correct Argument for Williams %R ---
                # 'ta' library uses 'lbp' (LookBack Period) instead of 'window' for this specific indicator
                df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'], lbp=14)
                
                # Momentum (ROC)
                df['momentum'] = ta.momentum.roc(df['close'], window=10)
                
                # SMA & EMA (20)
                df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
                df['ema_20'] = ta.trend.ema_indicator(df['close'], window=20)
                
                # Awesome Oscillator
                df['awesome_osc'] = ta.momentum.awesome_oscillator(df['high'], df['low'])

                # Grab the LATEST row (The most recent price)
                latest = df.iloc[-1]
                
                row = {
                    "symbol": symbol,
                    "price": latest["close"],
                    "rsi": latest["rsi"],
                    "macd": latest["macd"],
                    "macd_signal": latest["macd_signal"],
                    "williams_r": latest["williams_r"],
                    "momentum": latest["momentum"],
                    "awesome_osc": latest["awesome_osc"],
                    "sma_20": latest["sma_20"],
                    "ema_20": latest["ema_20"]
                }
                results.append(row)
                print(f"✅ Success: {symbol} | Price: {row['price']} | RSI: {row['rsi']:.2f}")

            except Exception as e:
                # This will print exact error if math fails again
                print(f"⚠️ Math Error on {symbol}: {e}")
        
        # 3. RESPECT RATE LIMIT
        print("💤 Sleeping 8 seconds...")
        time.sleep(8)

    # Save Final CSV
    if results:
        final_df = pd.DataFrame(results)
        # Ensure 'data/raw' folder exists
        output_dir = os.path.join(os.path.dirname(__file__), '../data/raw')
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, 'smart_technical_summary.csv')
        final_df.to_csv(output_path, index=False)
        print(f"\n💾 Data saved successfully to: {output_path}")

if __name__ == "__main__":
    main()