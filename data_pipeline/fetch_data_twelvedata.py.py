# improved_twelvedata_technical.py
import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise Exception("❌ API_KEY not found in .env file")

BASE_URL = "https://api.twelvedata.com"

TICKERS = [
    "EUR/USD", "GBP/USD", "USD/ZAR", "EUR/INR", "EUR/GBP",
    "XAU/USD", "XAG/USD",
    "BTC/USD", "ETH/USD", "SHIB/USD", "DOGE/USD",
    "SOL/USD", "LTC/USD", "XRP/USD",
    "SPX", "NVDA", "AAPL", "GOOG", "INTC", "NIO", "AMZN"
]

def safe_get_json(url, params=None, timeout=10):
    
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"❌ HTTP/JSON error for {url} params={params}: {e}")
        return None

def parse_indicator_value(item):
    
    if not isinstance(item, dict):
        return None
    for k, v in item.items():
        if k.lower() == "datetime":
            continue
        
        try:
            return float(v)
        except Exception:
            continue
    return None

def fetch_indicator(symbol, indicator, extra_params="", value_key=None, interval="1h", sleep=0.15):
    
    url = f"{BASE_URL}/{indicator}"
    params = {
        "symbol": symbol,
        "interval": interval,
        "apikey": API_KEY,
    }

    
    
    if extra_params:
        try:
            pairs = [p for p in extra_params.split("&") if p]
            for p in pairs:
                if "=" in p:
                    k, v = p.split("=", 1)
                    params[k] = v
        except Exception:
            pass

    json_data = safe_get_json(url, params=params)
    
    time.sleep(sleep)

    if not json_data:
        return None

    
    values = json_data.get("values") or json_data.get("value") or None
    if not values or not isinstance(values, list):
        
        
        return parse_indicator_value(json_data)

    first = values[0]
    if value_key:
        
        for kk, vv in first.items():
            if kk.lower() == value_key.lower():
                try:
                    return float(vv)
                except Exception:
                    return None
        return None

    
    return parse_indicator_value(first)



def signal_rsi(x):
    if x is None:
        return "NO_DATA"
    try:
        x = float(x)
    except:
        return "NO_DATA"
    if x > 70:
        return "SELL"
    if x < 30:
        return "BUY"
    return "NEUTRAL"

def signal_macd(x):
    if x is None:
        return "NO_DATA"
    try:
        x = float(x)
    except:
        return "NO_DATA"
    return "BUY" if x > 0 else "SELL"

def signal_ma(price, ma):
    if ma is None or price is None:
        return "NO_DATA"
    try:
        return "BUY" if float(price) > float(ma) else "SELL"
    except:
        return "NO_DATA"

def signal_cci(x):
    if x is None:
        return "NO_DATA"
    try:
        x = float(x)
    except:
        return "NO_DATA"
    if x > 100:
        return "BUY"
    if x < -100:
        return "SELL"
    return "NEUTRAL"


def fetch_full_technical_data(output_csv_relative="../data/raw/technical_summary.csv", interval="1h"):
    print("⬇️ Fetching ALL prices + indicators (improved & robust mode)...")

    
    base_dir = os.path.dirname(__file__)
    output_path = os.path.join(base_dir, output_csv_relative)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    
    symbol_str = ",".join(TICKERS)
    price_url = f"{BASE_URL}/price"
    price_json = safe_get_json(price_url, params={"symbol": symbol_str, "apikey": API_KEY})
    if price_json is None:
        price_json = {}

    results = []
    for symbol in TICKERS:
        
        price = None
        try:
            pinfo = price_json.get(symbol) if isinstance(price_json, dict) else None
            if pinfo and "price" in pinfo:
                price = float(pinfo["price"])
        except Exception:
            price = None

        print(f"\n📌 {symbol} (Fetching indicators…)")

        
        rsi = fetch_indicator(symbol, "rsi", interval=interval)
        
        macd = fetch_indicator(symbol, "macd", interval=interval, value_key="macd")
        sma = fetch_indicator(symbol, "sma", extra_params="&time_period=20", interval=interval)
        ema = fetch_indicator(symbol, "ema", extra_params="&time_period=20", interval=interval)
        cci = fetch_indicator(symbol, "cci", extra_params="&time_period=20", interval=interval)
        adx = fetch_indicator(symbol, "adx", extra_params="&time_period=14", interval=interval)
        williams = fetch_indicator(symbol, "williams", extra_params="&time_period=14", interval=interval, value_key="williams")
        
        stoch = fetch_indicator(symbol, "stochrsi", interval=interval, value_key="slowk")
        momentum = fetch_indicator(symbol, "momentum", extra_params="&time_period=20", interval=interval)
        ultimate = fetch_indicator(symbol, "ultimate", interval=interval)
        ppo = fetch_indicator(symbol, "ppo", interval=interval, value_key="ppo")
        awesome = fetch_indicator(symbol, "awesome", interval=interval)
        bull_bear = fetch_indicator(symbol, "bbpower", interval=interval, value_key="power")

        
        row = {
            "symbol": symbol,
            "price": price,

            "rsi": rsi,
            "rsi_signal": signal_rsi(rsi),

            "macd": macd,
            "macd_signal": signal_macd(macd),

            "sma20": sma,
            "sma20_signal": signal_ma(price, sma),

            "ema20": ema,
            "ema20_signal": signal_ma(price, ema),

            "cci": cci,
            "cci_signal": signal_cci(cci),

            "adx": adx,
            "williams_r": williams,
            "stochrsi_slowk": stoch,
            "momentum": momentum,
            "ultimate_osc": ultimate,
            "ppo": ppo,
            "awesome_osc": awesome,
            "bull_bear_power": bull_bear,
        }

        results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"\n💾 Saved: {output_path}")
    return df

if __name__ == "__main__":
    fetch_full_technical_data()
