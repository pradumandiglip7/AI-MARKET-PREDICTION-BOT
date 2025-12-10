import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv


load_dotenv()


API_KEY = os.getenv("API_KEY")

if not API_KEY:
    print("❌ Error: API_KEY not found in .env file")
    exit()

BASE_URL = "https://api.twelvedata.com"


TICKERS = [
    "EUR/USD", "GBP/USD", "USD/ZAR", "EUR/INR", "EUR/GBP",
    "XAU/USD", "XAG/USD",
    "BTC/USD", "ETH/USD", "SHIB/USD", "DOGE/USD", "SOL/USD", "LTC/USD", "XRP/USD",
    "SPX", "NVDA", "AAPL", "GOOG", "INTC", "NIO", "AMZN"
]

def fetch_all_prices():
    print("⬇️ Fetching prices from TwelveData...")
    results = []
    
    for symbol in TICKERS:
        try:
            url = f"{BASE_URL}/price?symbol={symbol}&apikey={API_KEY}"
            response = requests.get(url)
            data = response.json()
            
            if "price" in data:
                price = data["price"]
                print(f"✅ {symbol}: {price}")
                results.append({"symbol": symbol, "price": price})
            else:
                print(f"❌ {symbol}: No Data (Check API limit)")
            
            
            time.sleep(0.2) 
            
        except Exception as e:
            print(f"⚠️ Error fetching {symbol}: {e}")

     
    if results:
       
        output_path = os.path.join(os.path.dirname(__file__), '../data/raw/twelvedata_prices.csv')
        
        df = pd.DataFrame(results)
        df.to_csv(output_path, index=False)
        print(f"\n💾 Data saved successfully at: {output_path}")

if __name__ == "__main__":
    fetch_all_prices()