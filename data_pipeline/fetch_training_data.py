# # import os
# # import yfinance as yf
# # import pandas as pd
# # import ta  # Using the same library as your live bot
# # import time

# # # 1. SETUP: Yahoo Finance Symbols (Mapped from your list)
# # # Note: Yahoo uses '=X' for forex, '-USD' for crypto
# # TICKERS = [
# #     "EURUSD=X", "GBPUSD=X", "USDZAR=X", "EURINR=X", "EURGBP=X", # Forex
# #     "GC=F", "SI=F",                                             # Gold/Silver
# #     "BTC-USD", "ETH-USD", "SHIB-USD", "DOGE-USD", "SOL-USD", "LTC-USD", "XRP-USD", # Crypto
# #     "NVDA", "AAPL", "GOOG", "INTC", "NIO", "AMZN"               # Stocks
# # ]

# # # 2. CONFIGURATION
# # # Yahoo allows max ~730 days (2 years) for '1h' interval.
# # # For 10 years, you must use '1d' (Daily). 
# # # Let's start with 2 Years of Hourly data (Best for a bot).
# # PERIOD = "2y"    
# # INTERVAL = "1h" 

# # def fetch_and_calculate(symbol):
# #     print(f"⬇️ Downloading {PERIOD} of data for {symbol}...")
    
# #     # 1. Fetch History
# #     try:
# #         df = yf.download(symbol, period=PERIOD, interval=INTERVAL, progress=False)
        
# #         if df.empty:
# #             print(f"❌ No data found for {symbol}")
# #             return None
            
# #         # Fix: yfinance sometimes returns MultiIndex columns, we flatten them
# #         if isinstance(df.columns, pd.MultiIndex):
# #             df.columns = df.columns.get_level_values(0)
            
# #         # Clean column names (lowercase)
# #         df.columns = [c.lower() for c in df.columns]
        
# #         # Yahoo doesn't always have 'volume'. Fill 0 if missing.
# #         if 'volume' not in df.columns:
# #             df['volume'] = 0

# #         # 2. CALCULATE INDICATORS (Must match your Smart Fetcher EXACTLY)
# #         # RSI (14)
# #         df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        
# #         # MACD (12, 26, 9)
# #         macd_obj = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
# #         df['macd'] = macd_obj.macd()
# #         df['macd_signal'] = macd_obj.macd_signal()
        
# #         # Williams %R (14) - using 'lbp' as we learned
# #         df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'], lbp=14)
        
# #         # Momentum (ROC - 10)
# #         df['momentum'] = ta.momentum.roc(df['close'], window=10)
        
# #         # SMA & EMA (20)
# #         df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
# #         df['ema_20'] = ta.trend.ema_indicator(df['close'], window=20)
        
# #         # Awesome Oscillator
# #         df['awesome_osc'] = ta.momentum.awesome_oscillator(df['high'], df['low'])
        
# #         # 3. CREATE TARGET (The "Answer Key" for the AI)
# #         # We want the AI to predict if Price goes UP next hour.
# #         # Shift(-1) means "Look at the NEXT row"
# #         df['future_close'] = df['close'].shift(-1)
        
# #         # Target = 1 if Next Price > Current Price, else 0
# #         df['target'] = (df['future_close'] > df['close']).astype(int)
        
# #         # Drop NaN values (the first 26 rows of calculations + last row with no future)
# #         df.dropna(inplace=True)
        
# #         # Add symbol column so we know which stock this is
# #         df['symbol'] = symbol
        
# #         return df

# #     except Exception as e:
# #         print(f"⚠️ Error on {symbol}: {e}")
# #         return None

# # def main():
# #     all_data = []
    
# #     print("🚀 Starting Historical Data Mining (Yahoo Finance)...")
    
# #     for ticker in TICKERS:
# #         df = fetch_and_calculate(ticker)
# #         if df is not None:
# #             all_data.append(df)
# #             print(f"✅ Processed {len(df)} rows for {ticker}")
        
# #         # Be polite to Yahoo API
# #         time.sleep(1)

# #     # Combine everything into one massive file
# #     if all_data:
# #         full_df = pd.concat(all_data)
        
# #         # Save to a new folder for training
# #         output_dir = os.path.join(os.path.dirname(__file__), '../data/training')
# #         os.makedirs(output_dir, exist_ok=True)
        
# #         file_path = os.path.join(output_dir, 'full_history_data.csv')
# #         full_df.to_csv(file_path)
        
# #         print(f"\n🎉 DONE! Saved {len(full_df)} rows of training data.")
# #         print(f"📂 File location: {file_path}")
# #     else:
# #         print("❌ Failed to get any data.")

# # if __name__ == "__main__":
# #     main()




# import os
# import yfinance as yf
# import pandas as pd
# import ta
# import time

# # 1. SETUP
# TICKERS = [
#     "EURUSD=X", "GBPUSD=X", "USDZAR=X", "EURINR=X", "EURGBP=X", 
#     "GC=F", "SI=F",                                             
#     "BTC-USD", "ETH-USD", "SHIB-USD", "DOGE-USD", "SOL-USD", "LTC-USD", "XRP-USD", 
#     "NVDA", "AAPL", "GOOG", "INTC", "NIO", "AMZN"               
# ]

# # --- CHANGE IS HERE ---
# PERIOD = "5y"    # 5 Years of history
# INTERVAL = "1d"  # Daily candles (Required for >2 years data)

# def fetch_and_calculate(symbol):
#     print(f"⬇️ Downloading {PERIOD} of data for {symbol}...")
    
#     try:
#         df = yf.download(symbol, period=PERIOD, interval=INTERVAL, progress=False)
        
#         if df.empty:
#             print(f"❌ No data found for {symbol}")
#             return None
            
#         # Fix MultiIndex columns if present
#         if isinstance(df.columns, pd.MultiIndex):
#             df.columns = df.columns.get_level_values(0)
            
#         df.columns = [c.lower() for c in df.columns]
        
#         if 'volume' not in df.columns:
#             df['volume'] = 0

#         # 2. CALCULATE INDICATORS
#         # Note: These are now calculated on DAILY candles
#         df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        
#         macd_obj = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
#         df['macd'] = macd_obj.macd()
#         df['macd_signal'] = macd_obj.macd_signal()
        
#         df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'], lbp=14)
#         df['momentum'] = ta.momentum.roc(df['close'], window=10)
        
#         df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
#         df['ema_20'] = ta.trend.ema_indicator(df['close'], window=20)
#         df['awesome_osc'] = ta.momentum.awesome_oscillator(df['high'], df['low'])
        
#         # 3. CREATE TARGET
#         # We predict if TOMORROW'S Close is higher than TODAY'S Close
#         df['future_close'] = df['close'].shift(-1)
#         df['target'] = (df['future_close'] > df['close']).astype(int)
        
#         df.dropna(inplace=True)
#         df['symbol'] = symbol
        
#         return df

#     except Exception as e:
#         print(f"⚠️ Error on {symbol}: {e}")
#         return None

# def main():
#     all_data = []
#     print(f"🚀 Starting Historical Data Mining ({PERIOD} / {INTERVAL})...")
    
#     for ticker in TICKERS:
#         df = fetch_and_calculate(ticker)
#         if df is not None:
#             all_data.append(df)
#             print(f"✅ Processed {len(df)} rows for {ticker}")
#         time.sleep(1)

#     if all_data:
#         full_df = pd.concat(all_data)
        
#         output_dir = os.path.join(os.path.dirname(__file__), '../data/training')
#         os.makedirs(output_dir, exist_ok=True)
        
#         file_path = os.path.join(output_dir, 'full_history_data.csv')
#         full_df.to_csv(file_path)
        
#         print(f"\n🎉 DONE! Saved {len(full_df)} rows of training data.")
#         print(f"📂 File location: {file_path}")
#     else:
#         print("❌ Failed to get any data.")

# if __name__ == "__main__":
#     main()





# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------





import os
import yfinance as yf
import pandas as pd
import ta
import time

# TUMHARI LIST
TICKERS = [
    "EURUSD=X", "GBPUSD=X", "USDZAR=X", "BTC-USD", "ETH-USD", 
    "XRP-USD", "NVDA", "AAPL", "GOOG", "AMZN"
]
0
# --- CHANGE 1: Config for Short Term ---
TIMEFRAMES = [
    {"period": "60d", "interval": "5m", "save_name": "data_5m.csv"},
    {"period": "60d", "interval": "15m", "save_name": "data_15m.csv"}
]

def fetch_and_process(symbol, period, interval):
    try:
        # Download Data
        df = yf.download(symbol, period=period, interval=interval, progress=False)
        if df.empty: return None
        
        # Flatten MultiIndex (Fix for Yahoo update)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        # Clean columns
        df.columns = [c.lower() for c in df.columns]
        if 'volume' not in df.columns: df['volume'] = 0

        # --- CALCULATE INDICATORS (Wahi Same Math) ---
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        macd = ta.trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'], lbp=14)
        df['momentum'] = ta.momentum.roc(df['close'], window=10)
        df['ema_20'] = ta.trend.ema_indicator(df['close'], window=20)
        
        # --- TARGET ---
        # 5m data me hum agle 5m ki prediction chahte hain
        df['future_close'] = df['close'].shift(-1)
        df['target'] = (df['future_close'] > df['close']).astype(int)
        
        df.dropna(inplace=True)
        df['symbol'] = symbol
        return df
        
    except Exception as e:
        print(f"⚠️ Error {symbol}: {e}")
        return None

def main():
    # Loop through each timeframe (5m, 15m)
    for tf_config in TIMEFRAMES:
        all_data = []
        p, i, fname = tf_config["period"], tf_config["interval"], tf_config["save_name"]
        
        print(f"\n🚀 Fetching {i} Data (Last {p})...")
        
        for ticker in TICKERS:
            df = fetch_and_process(ticker, p, i)
            if df is not None:
                all_data.append(df)
                print(f"✅ {ticker}: {len(df)} rows")
            time.sleep(1) # Yahoo ko gussa na dilayein
            
        if all_data:
            full_df = pd.concat(all_data)
            path = f"data/training/{fname}"
            full_df.to_csv(path)
            print(f"💾 Saved: {path}")

if __name__ == "__main__":
    main()