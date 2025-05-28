import requests
import time
import hmac
import hashlib
from datetime import datetime, timezone
import numpy as np
import pandas as pd

# === PLACEHOLDERS ===
TELEGRAM_TOKEN = "7716430771:AAHqCZNoDACm3qlaue4G_hTJkyrxDRV9uxo"
TELEGRAM_CHAT_ID = "6487259893"
API_KEY = 'mx0vglybt9Fm3utSdv'
API_SECRET = 'e8b636b031c2451d91ad03b4928b18bd'

# --- Constants ---
PRICE_CHANGE_THRESHOLD = 10 # in percent
RSI_THRESHOLD = 70
RSI_PERIOD = 14
EMA_LONG_PERIOD = 200
        
def get_perpetual_symbols():
    url = "https://contract.mexc.com/api/v1/contract/detail"
    res = requests.get(url).json()
    return [s["symbol"] for s in res["data"] if s["quoteCoin"] == "USDT"]

def get_open_symbols(market_type="spot"):
 if market_type == "spot":
    url = "https://api.mexc.com/api/v3/account"
    timestamp = int(time.time() * 1000)
    query_string = f"timestamp={timestamp}"
    signature = hmac.new(
        API_SECRET.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    headers = {
        "X-MEXC-APIKEY": API_KEY
    }
    full_url = f"{url}?{query_string}&signature={signature}"
    response = requests.get(full_url, headers=headers)
    if response.status_code != 200:
        print("Spot balance fetch failed:", response.status_code)
        return []
    balances = response.json().get("balances", [])
    return [b["asset"] + "USDT" for b in balances if float(b["free"]) + float(b["locked"]) > 0]    
 elif market_type == "futures":
    url = "https://contract.mexc.com/api/v1/private/position/open_positions"
    timestamp = str(int(time.time() * 1000))
    params = {
        "api_key": API_KEY,
        "req_time": timestamp,
    }
    query_string = '&'.join([f"{key}={params[key]}" for key in sorted(params)])
    signature = hmac.new(
        API_SECRET.encode('utf-8'),
        query_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    params["sign"] = signature
    headers = {
        "Content-Type": "application/x-www-form-urlencoded"
    }
    response = requests.post(url, data=params, headers=headers)
    if response.status_code != 200:
        print("Futures position fetch failed:", response.status_code)
        return []
    data = response.json().get("data", [])
    return[item["symbol"] for item in data if float(item.get("holdVol", 0)) > 0]
 else:
    print("Invalid market_type. Use 'spot' or 'futures'.")
    return []

def get_candles(symbol, market_type="spot", interval="4H", limit= EMA_LONG_PERIOD + 1):
 if market_type == "spot":
    if interval == "4H":
         interval = "4h"
    elif interval == "1D":
         interval = "1d"
    url = f"https://api.mexc.com/api/v3/klines"
    params = {'symbol': symbol, 'interval': interval, 'limit': limit}
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"Failed to fetch spot candles for {symbol}: {response.text}")
        return []
    data = response.json()
    if not data or len(data[0]) < 6:
        print(f"Unexpected spot candle structure for {symbol}")
        return []
    candles = [
        (int(item[0]), float(item[1]), float(item[2]), float(item[3]), float(item[4]), float(item[5]))
        for item in data
    ]
    return candles        
 elif market_type == "futures":
    if interval == "4H":
         interval = "Hour4"
    elif interval == "1D":
         interval = "Day1"
    url = f"https://contract.mexc.com/api/v1/contract/kline/{symbol}"
    params = {'interval': interval, 'limit': limit}
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"Failed to fetch candles for {symbol}")
        return []
    result = response.json()
    data = result.get('data', {})
    # We'll zip the columns into OHLC-style rows
    if not data or not all(k in data for k in ['time', 'close']):
        print(f"Unexpected candle structure for {symbol}")
        return []
    candles = list(zip(
        data['time'],
        data['open'],
        data['high'],
        data['low'],
        data['close'],
        data['vol']
    ))
    return candles
 else:
    print("Invalid market_type. Use 'spot' or 'futures'.")
    return []
        
def get_12h_candles_from_4h(candles_4h, market_type="spot"):
 if market_type == "spot":
    if len(candles_4h) < 3:
        return []
    candles_12h = []
    # Group each 3x 4H candles into one 12H candle
    for i in range(0, len(candles_4h) - 2, 3):
        c1 = candles_4h[i]
        c2 = candles_4h[i + 1]
        c3 = candles_4h[i + 2]

        t_open = c1[0]                          # Timestamp of first 4H candle
        o = c1[1]                               # Open of first candle
        h = max(c1[2], c2[2], c3[2])            # Highest high
        l = min(c1[3], c2[3], c3[3])            # Lowest low
        c = c3[4]                               # Close of last 4H candle
        v = c1[5] + c2[5] + c3[5]               # Combined volume
        candles_12h.append((t_open, o, h, l, c, v))
    return candles_12h
 elif market_type == "futures":
    candles_12h = []
    for i in range(0, len(candles_4h) - 2, 3):
        group = candles_4h[i:i + 3]
        if len(group) < 3:
            continue
        times = [g[0] for g in group]
        opens = [float(g[1]) for g in group]
        highs = [float(g[2]) for g in group]
        lows = [float(g[3]) for g in group]
        closes = [float(g[4]) for g in group]
        volumes = [float(g[5]) for g in group]
        candle_12h = (
            times[0],             # timestamp of first 4H candle
            opens[0],             # open of first
            max(highs),           # high of group
            min(lows),            # low of group
            closes[-1],           # close of last
            sum(volumes)          # sum of volume
        )
        candles_12h.append(candle_12h)
    return candles_12h
 else:
        print("Invalid market_type. Use 'spot' or 'futures'.")
        return []

def calculate_rsi(closes, period=14):
    if len(closes) < period + 1:
        return None
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(closes, fast_period=12, slow_period=26, signal_period=9):
    closes = pd.Series(closes)  # Ensure it's a Series
    if len(closes) < slow_period + signal_period:
        return None, None, None

    ema_fast = closes.ewm(span=fast_period, adjust=False).mean()
    ema_slow = closes.ewm(span=slow_period, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    macd_histogram = macd_line - signal_line

    return macd_line.iloc[-1], signal_line.iloc[-1], macd_histogram.iloc[-1]

def calculate_ema(closes, period=200):
    closes = pd.Series(closes)  # Ensure it's a Series
    if len(closes) < period:
        return None
    return pd.Series(closes).ewm(span=period, adjust=False).mean().iloc[-1]

def touch_ema_200(candles, closes_12h, closes_1d):
        if len(candles) < 3:
            return False

        t, o, h, l, c, v = candles[-1]
        h, l = float(h), float(l)

        if len(closes_12h) > 200:
            ema_200_12h = calculate_ema(closes_12h)
            if ema_200_12h > l and ema_200_12h < h:
                return True
         
        if len(closes_1d) > 200:
            ema_200_1d = calculate_ema(closes_1d)
            if ema_200_1d > l and ema_200_1d < h:
                return True

def calculate_stoch_rsi(closes, rsi_period=14, stoch_period=14, smooth_k=3, smooth_d=3):
    if len(closes) < rsi_period + stoch_period:
        return None, None
    rsi_series = pd.Series([calculate_rsi(closes[i - rsi_period:i + 1], rsi_period) for i in range(rsi_period, len(closes))])
    stoch_rsi = (rsi_series - rsi_series.rolling(stoch_period).min()) / (rsi_series.rolling(stoch_period).max() - rsi_series.rolling(stoch_period).min())
    k = stoch_rsi.rolling(smooth_k).mean()
    d = k.rolling(smooth_d).mean()
    return k.iloc[-1] * 100, d.iloc[-1] * 100

def detect_candle_patterns(candles, pattern_name="4H"):
    if len(candles) < 3:
        return ""
    messages = []

    # We'll use the last two candles to detect engulfing patterns
    prev = candles[-3]
    curr = candles[-2]

    t1, o1, h1, l1, c1, v1 = prev
    t2, o2, h2, l2, c2, v2 = curr
    o1, c1 = float(o1), float(c1)
    o2, c2 = float(o2), float(c2)
    h2, l2 = float(h2), float(l2)

    # Bullish Engulfing: previous candle is red, current is green and body engulfs
    if c1 < o1 and c2 > o2 and o2 < c1 and c2 > o1:
        messages.append(f"🟢 Bullish Engulfing on {pattern_name}")
    # Bearish Engulfing: previous candle is green, current is red and body engulfs
    elif c1 > o1 and c2 < o2 and o2 > c1 and c2 < o1:
        messages.append(f"🔴 Bearish Engulfing on {pattern_name}")

    # Continue checking other patterns for the last candle
    o, h, l, c = o2, h2, l2, c2
    body = abs(c - o)
    upper_wick = h - max(c, o)
    lower_wick = min(c, o) - l
    total_range = h - l
    #d1_2_body = total_range/2 + l
    d1_3_body = total_range/3 + l
    d3_4_body = h - total_range/3
    f61_8_boddy = h - total_range * 61.8
    f38_2_boddy = l + total_range * 38.2


    if total_range == 0:
        return "\n".join(messages)

    #body_ratio = body / total_range
    #upper_ratio = upper_wick / total_range
    #lower_ratio = lower_wick / total_range

    # Hammer
    #if lower_ratio > 0.6 and upper_ratio < 0.2 and body_ratio < 0.3:
    if min(c, o) > f61_8_boddy :
        messages.append(f"🔨 Hammer detected on {pattern_name}")
    # Inverted Hammer
    elif max(c, o) < f38_2_boddy :
        messages.append(f"🔻 Inverted Hammer on {pattern_name}")
    # Spinning Top
    elif min(c, o) >  d1_3_body and max(c, o) < d3_4_body:
        messages.append(f"🌀 Spinning Top on {pattern_name}")

    return "\n".join(messages)
    
def send_telegram_alert(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }

    try:
        response = requests.post(url, data=data)
        if response.status_code != 200:
            print(f"Failed to send Telegram alert: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Error sending Telegram alert: {e}")

def main():
    now = datetime.now(timezone.utc)
    hour, minute = now.hour, now.minute
    symbols = [ "BTC_USDT", "ETH_USDT", "ADA_USDT", "SOL_USDT" ]
    sym_spots = [ "JASMYUSDT", "FARTCOINUSDT", "KASUSDT", "COOKIEUSDT", "RIZUSDT", "POPCATUSDT", "PIPPINUSDT" ]
    #sym_spots = get_open_symbols("spot")
    #sym_futs = get_open_symbols("futures")
    print(f"{sym_spots}")
    #print(f"{sym_futs}")    
    #symbols = get_perpetual_symbols()

    for sym_spot in sym_spots:
        candles_4h = get_candles(sym_spot,"spot",interval="4H",limit=12)
        print(f"{sym_spot} {candles_4h}")
        candles_12h = get_12h_candles_from_4h(candles_4h)
        #print(f"{sym_spot} {candles_12h}")
        candles_1d = get_candles(sym_spot,"spot",interval="1D",limit=3)
        print(f"{sym_spot} {candles_1d}")
        
        if len(candles_4h) < 12:
            continue 
        
        candelsticks_msg = ""
        #if hour in [0, 4, 8, 12, 16, 20]:
        candelsticks_msg = detect_candle_patterns(candles_4h, "4H")
        #if hour in [0, 12]:
        candelsticks_msg += detect_candle_patterns(candles_12h, "12H")
       # if hour == 0:
        candelsticks_msg += detect_candle_patterns(candles_1d, "1D")
        if candelsticks_msg:
           print(f"{sym_spot} \n{candelsticks_msg}")
           #send_telegram_alert(candelsticks_msg)
        
    for symbol in symbols:
        #candles_5m = get_candles(symbol, interval='Min5',limit=3)
        #candles_15m = get_candles(symbol, interval='Min15',limit=3)
        candles_1h = get_candles(symbol, "futures",interval='Min60')
        candles_4h = get_candles(symbol,"futures",interval="4H",limit=(EMA_LONG_PERIOD * 3))
        candles_12h = get_12h_candles_from_4h(candles_4h)
        candles_1d = get_candles(symbol,"futures",interval="1D")
        candles_1W = get_candles(symbol,"futures",interval='Week1')
        candles_1M = get_candles(symbol,"futures",interval='Month1')

        if len(candles_4h) < 14:
            continue 
        
        closes_1h = [float(c[4]) for c in candles_4h]
        closes_4h = [float(c[4]) for c in candles_4h]
        closes_12h = [float(c[4]) for c in candles_12h]
        closes_1d = [float(c[4]) for c in candles_1d]
        closes_1W = [float(c[4]) for c in candles_1W]
        closes_1M = [float(c[4]) for c in candles_1M]

         #Candelsticks pattern erkennung
        candelsticks_msg = ""
        #if hour in [0, 4, 8, 12, 16, 20]:
        candelsticks_msg = detect_candle_patterns(candles_4h, "4H")
        #if hour in [0, 12]:
        candelsticks_msg += detect_candle_patterns(candles_12h, "12H")
       # if hour == 0:
        candelsticks_msg += detect_candle_patterns(candles_1d, "1D")
        if candelsticks_msg:
                #candelsticks_msg += candelsticks_msg
           print(f"{symbol} \n{candelsticks_msg}")
           #send_telegram_alert(candelsticks_msg)

        change_pct_1d = 0
        change_pct_1W = 0
        change_pct_1M = 0
        change_pct_4h = ((closes_4h[-1] - closes_4h[-2]) / closes_4h[-2]) * 100
        #print(f"{symbol} {closes_4h[-1]} {closes_4h[-2]}") 
        if len(closes_1d) > 2: 
            change_pct_1d = ((closes_1d[-1] - closes_1d[-2]) / closes_1d[-2]) * 100
        if len(closes_1W) > 2: 
            change_pct_1W = ((closes_1W[-1] - closes_1W[-2]) / closes_1W[-2]) * 100
        if len(closes_1M) > 2:
            change_pct_1M = ((closes_1M[-1] - closes_1M[-2]) / closes_1M[-2]) * 100

        rsi_4h = 0
        rsi_1d = 0
        if len(candles_4h) > 14:
            rsi_4h = calculate_rsi(closes_4h)
        if len(candles_1d) > 14:
            rsi_1d = calculate_rsi(closes_1d)

        macd_4h_condition = False
        macd_1d_condition = False
        if len(candles_4h) >= 50:
            macd_4h, signal_4h, hist_4h = calculate_macd(closes_4h)
            macd_4h_condition = macd_4h > 0 and signal_4h > 0 and hist_4h > 0
        if len(candles_1d) >= 50:
            macd_1d, signal_1d, hist_1d = calculate_macd(closes_1d)
            macd_1d_condition = macd_1d > 0 and signal_1d > 0 and hist_1d > 0
            #print(f"{symbol} MACD: {macd_1d:.3f}, Signal: {signal_1d:.3f}, Histogram: {hist_1d:.3f}")

        is_touch_ema_200 = touch_ema_200(candles_1h,closes_12h,closes_1d)
                
        if change_pct_4h > PRICE_CHANGE_THRESHOLD and rsi_4h and rsi_4h > RSI_THRESHOLD:
            #print(f"{symbol}")
            message = f"🚨 {symbol}\n4h:{change_pct_4h:.2f}% rsi:{rsi_4h:.2f} macd:{macd_4h_condition}\n1D:{change_pct_1d:.2f}% rsi:{rsi_1d:.2f} macd:{macd_1d_condition}\nema200:{is_touch_ema_200} W:{change_pct_1W:.2f}% M:{change_pct_1M:.2f}%\n"
            #send_telegram_alert(message)

if __name__ == "__main__":
    main()
