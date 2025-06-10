import requests
import time
import hmac
import hashlib
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import csv
import os

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
        
def get_all_perpetual_symbols():
    url = "https://contract.mexc.com/api/v1/contract/detail"
    res = requests.get(url).json()
    return [s["symbol"] for s in res["data"] if s["quoteCoin"] == "USDT"]
        
def load_watchlist_from_csv(file_path):
    symbols = []
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return []
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            for cell in row:
                symbol = cell.strip()
                if symbol.startswith("MEXC:") or symbol.startswith("BITGET:"):
                   symbol = symbol.replace("MEXC:", "")
                   symbol = symbol.replace("BITGET:", "")
                if symbol.endswith(".P"):
                   symbol = symbol.replace(".P", "").replace("USDT", "") + "_USDT"
                symbols.append(symbol)
    return symbols

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
    #return [b["asset"] + "USDT" for b in balances if float(b["free"]) + float(b["locked"]) > 0] 
    open_assets = [] 
    for b in balances:
        asset = b["asset"]
        free = float(b.get("free", 0))
        locked = float(b.get("locked", 0))
        if free + locked > 0 and asset != "USDT":
           open_assets.append(asset + "USDT")
    return open_assets
 elif market_type == "futures":
    url = "https://contract.mexc.com/api/v1/private/position/open_positions"
    timestamp = str(int(time.time() * 1000))
    object_string = API_KEY + timestamp
    signature = hmac.new(
        API_SECRET.encode('utf-8'),
        object_string.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()
    headers = {
        "ApiKey": API_KEY,
        "Request-Time": timestamp,
        "Signature": signature,
        "Content-Type": "application/json"
    }
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print("Futures request failed:", response.status_code, response.text)
        return []
    data = response.json().get("data", [])
    return [item["symbol"] for item in data if float(item.get("holdVol", 0)) > 0]         
 else:
    print("Invalid market_type. Use 'spot' or 'futures'.")
    return []

def get_candles(symbol, market_type, interval, limit= EMA_LONG_PERIOD + 1):
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
        
def get_12h_candles_from_4h(candles_4h):
    if len(candles_4h) < 4:
        return []

    # Remove most recent (potentially forming) candle
    closed_candles = candles_4h[:-1]
    if len(closed_candles) < 3:
        return []

    # Determine how many candles to remove based on the second candle's hour
    timestamp = int(closed_candles[1][0])
    if timestamp > 9999999999:
        timestamp = timestamp // 1000
    hour = datetime.utcfromtimestamp(timestamp).hour

    # Custom logic based on your rules
    remove_map = {
        0: 1,
        4: 2,
        8: 0,
        12: 1,
        16: 2,
        20: 0
    }

    remove_count = remove_map.get(hour, None)
    if remove_count is None or len(closed_candles) <= remove_count:
        return []

    aligned_candles = closed_candles[remove_count:]

    # Build 12H candles from 3x 4H groups
    candles_12h = []
    for i in range(0, len(aligned_candles) - 2, 3):
        group = aligned_candles[i:i + 3]
        if len(group) != 3:
            continue

        t = group[0][0]
        o = float(group[0][1])
        h = max(float(c[2]) for c in group)
        l = min(float(c[3]) for c in group)
        c = float(group[2][4])
        v = sum(float(c[5]) for c in group)
        #print(f"({t},{o},{h},{l},{c},{v})")
        candles_12h.append((t, o, h, l, c, v))

    return candles_12h

def calculate_ema(closes, period=200):
    closes = pd.Series(closes)  # Ensure it's a Series
    if len(closes) < period:
        return None
    return pd.Series(closes).ewm(span=period, adjust=False).mean().iloc[-1]

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

def calculate_stoch_rsi(closes, rsi_period=14, stoch_period=14, smooth_k=3, smooth_d=3):
    if len(closes) < rsi_period + stoch_period:
        return None, None
    rsi_series = pd.Series([calculate_rsi(closes[i - rsi_period:i + 1], rsi_period) for i in range(rsi_period, len(closes))])
    stoch_rsi = (rsi_series - rsi_series.rolling(stoch_period).min()) / (rsi_series.rolling(stoch_period).max() - rsi_series.rolling(stoch_period).min())
    k = stoch_rsi.rolling(smooth_k).mean()
    d = k.rolling(smooth_d).mean()
    return k.iloc[-1] * 100, d.iloc[-1] * 100

def calculate_ichimoku(candles):
    if len(candles) < 52:
        return None, None, None, None  # Not enough data

    highs = pd.Series([float(c[2]) for c in candles])
    lows = pd.Series([float(c[3]) for c in candles])
    
    # Tenkan-sen (Conversion Line)
    nine_high = highs.rolling(window=9).max()
    nine_low = lows.rolling(window=9).min()
    tenkan_sen = (nine_high + nine_low) / 2

    # Kijun-sen (Base Line)
    period26_high = highs.rolling(window=26).max()
    period26_low = lows.rolling(window=26).min()
    kijun_sen = (period26_high + period26_low) / 2

    # Senkou Span A (Leading Span A)
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)

    # Senkou Span B (Leading Span B)
    period52_high = highs.rolling(window=52).max()
    period52_low = lows.rolling(window=52).min()
    senkou_span_b = ((period52_high + period52_low) / 2).shift(26)

    return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b

def alarm_touch_ema_200(symbol, candles_4h, candles_12h, candles_1d, priority):

    if len(candles_12h) < 200:
       return 
    t, o, h, l, c, v = candles_4h[-1]
    h, l = float(h), float(l)
    closes_12h = [float(c[4]) for c in candles_12h]
    ema_200_12h = calculate_ema(closes_12h)
    print(f"{symbol}ema{ema_200_12h}:h{h}l{l}")
    if ema_200_12h > l and ema_200_12h < h:
       send_telegram_alert(symbol, 'touched Ema200_12H', priority)
    if len(candles_1d) < 200:
       return 
    closes_1d = [float(c[4]) for c in candles_1d]        
    ema_200_1d = calculate_ema(closes_1d)
    print(f"{symbol}ema{ema_200_1d}:h{h}l{l}")
    if ema_200_1d > l and ema_200_1d < h:
       send_telegram_alert(symbol, 'touched Ema200_1d', priority)

def alarm_candle_patterns(symbol, candles, pattern_name, priority):
    if len(candles) < 3:
        return
    messages = []
    pos = 2 if (pattern_name == "12H") else 3
    pos_0 = pos-1
    # We'll use the last two candles to detect engulfing patterns
    prev = candles[-pos]
    curr = candles[-pos_0]

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
    f61_8_boddy = h - total_range * 0.618
    f38_2_boddy = l + total_range * 0.382
    # Fibonacci 61.8% level
    fib_618 = l + 0.618 * total_range
    corp_top = max(o, c)
    corp_bottom = min(o, c)

    if total_range == 0:
        return "\n".join(messages)

    body_ratio = body / total_range
    upper_ratio = upper_wick / total_range
    lower_ratio = lower_wick / total_range
    print(f"{symbol}, (o {o}, h{h},l{l},c{c}) - (bd{body_ratio},up{upper_ratio},lo{lower_ratio})")
    # Hammer
    if lower_ratio > 0.6 and upper_ratio < 0.2 and body_ratio < 0.3:
    #if lower_ratio > 0.6 and upper_ratio < 0.2 and body_ratio < 0.3 and corp_bottom >= fib_618:
    #if min(c, o) > f61_8_boddy :
        messages.append(f"🔨 Hammer detected on {pattern_name}")
    # Inverted Hammer
    #elif max(c, o) < f38_2_boddy :
    #elif upper_ratio > 0.6 and lower_ratio < 0.2 and body_ratio < 0.3 and corp_top <= fib_618:
    elif upper_ratio > 0.6 and lower_ratio < 0.2 and body_ratio < 0.3:
        messages.append(f"🔻 Inverted Hammer on {pattern_name}")
    # Spinning Top
    #elif min(c, o) >  d1_3_body and max(c, o) < d3_4_body:
    elif body_ratio < 0.3 and upper_ratio > 0.3 and lower_ratio > 0.3:
        messages.append(f"🌀 Spinning Top on {pattern_name}")
         
    if messages:
       messages = "\n".join(messages)   
       #print(f"[{symbol}]{messages}")
       send_telegram_alert(symbol, messages, priority)
 
def alarm_ichimoku_crosses(symbol, candles, tf_label="", priority=False):
    if len(candles) < 52:
        return ""
    
    closes = [float(c[4]) for c in candles]
    current_close = closes[-1]

    messages = []

    tenkan, kijun, senkou_a, senkou_b = calculate_ichimoku(candles)
    # Ichimoku Cloud boundaries
    latest_senkou_a = senkou_a.iloc[-27] if len(senkou_a) >= 27 else None
    latest_senkou_b = senkou_b.iloc[-27] if len(senkou_b) >= 27 else None
    if latest_senkou_a is None or latest_senkou_b is None:
        return ""

    cloud_top = max(latest_senkou_a, latest_senkou_b)
    cloud_bottom = min(latest_senkou_a, latest_senkou_b)

    # Tenkan/Kijun Cross
    print(f"[{symbol}]ichimoku -2({tenkan.iloc[-2]},{kijun.iloc[-2]}),-1({tenkan.iloc[-1]},{kijun.iloc[-1]}), senk ({senkou_a.iloc[-27]},{senkou_b.iloc[-27]})")
    if tenkan.iloc[-2] < kijun.iloc[-2] and tenkan.iloc[-1] >= kijun.iloc[-1]:
        if current_close > cloud_top:
            messages.append(f"🟢 Bullish Tenkan/Kijun cross above cloud on {tf_label}")
        else:
            messages.append(f"🟡 Bullish Tenkan/Kijun cross below/inside cloud on {tf_label}")
    elif tenkan.iloc[-2] > kijun.iloc[-2] and tenkan.iloc[-1] <= kijun.iloc[-1]:
        if current_close < cloud_bottom:
            messages.append(f"🔴 Bearish Tenkan/Kijun cross below cloud on {tf_label}")
        else:
            messages.append(f"🟠 Bearish Tenkan/Kijun cross above/inside cloud on {tf_label}")

    # Senkou Span A/B Cross (Cloud twist)
    #if senkou_a.iloc[-2] < senkou_b.iloc[-2] and senkou_a.iloc[-1] > senkou_b.iloc[-1]:
    #    messages.append(f"🟢 Bullish Cloud Twist (Span A > B) on {tf_label}")
    #elif senkou_a.iloc[-2] > senkou_b.iloc[-2] and senkou_a.iloc[-1] < senkou_b.iloc[-1]:
     #   messages.append(f"🔴 Bearish Cloud Twist (Span A < B) on {tf_label}")
       
    if messages:
       messages = "\n".join(messages)
       send_telegram_alert(symbol, messages, priority)
                       
def send_telegram_alert(symbol, message, priority):
    if "_" in symbol:
       symbol = symbol.replace("_USDT", "USDT.P")
    prefix = "🚨" if priority else ""
    message = f"{prefix}[{symbol}](https://www.tradingview.com/chart/?symbol=MEXC:{symbol})\n{message}"
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    data = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown",
        "disable_web_page_preview": True
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

    open_spots = get_open_symbols("spot")
    #open_spots = []    
    for open_spot in open_spots:
        candles_4h = get_candles(open_spot, "spot",interval="4H",limit=601)
        candles_12h = get_12h_candles_from_4h(candles_4h)
        print(f"{open_spot}4h:{candles_4h[-1]}12h{candles_12h[-1]}")
        alarm_candle_patterns(open_spot, candles_12h, "12H", True)
        candles_1d = get_candles(open_spot,"spot",interval="1D")     
        closes_4h = [float(c[4]) for c in candles_4h]
        stoch_rsiK, stoch_rsiD = calculate_stoch_rsi(closes_4h)
        if stoch_rsiK and (stoch_rsiK < 20 or stoch_rsiK > 80): 
           alarm_candle_patterns(open_spot, candles_4h, "4H", True)
           if hour in [0, 12]:
              alarm_candle_patterns(open_spot, candles_12h, "12H", True)
           if hour == 0:
              alarm_candle_patterns(open_spot, candles_1d, "1D", True)
                
    open_futures = get_open_symbols("futures")
    #open_futures = []
    for open_future in open_futures:
        candles_4h = get_candles(open_future, "futures",interval="4H",limit=601)
        candles_12h = get_12h_candles_from_4h(candles_4h)
        candles_1d = get_candles(open_future,"futures",interval="1D")     
        closes_4h = [float(c[4]) for c in candles_4h]
        stoch_rsiK, stoch_rsiD = calculate_stoch_rsi(closes_4h)
        if stoch_rsiK and (stoch_rsiK < 20 or stoch_rsiK > 80): 
           alarm_candle_patterns(open_future, candles_4h, "4H", True)
           if hour in [0, 12]:
              alarm_candle_patterns(open_future, candles_12h, "12H", True)
           if hour == 0:
              alarm_candle_patterns(open_future, candles_1d, "1D", True)
        
    #watchlist_symbols = load_watchlist_from_csv("watchlists/Shorts.csv")
    #alarm_candle_patterns(watchlist_symbols, 'futures', False)
        
    #allf_symbols = get_all_perpetual_symbols()
    allf_symbols = ["L3_USDT", "AXL_USDT" ]    
    for allf_symbol in allf_symbols:
        candles_4h = get_candles(allf_symbol, "futures",interval="4H",limit=601)
        candles_12h = get_12h_candles_from_4h(candles_4h)
        candles_1d = get_candles(allf_symbol,"futures",interval="1D",limit=601)
        alarm_touch_ema_200(allf_symbol, candles_4h, candles_12h, candles_1d, False)
        alarm_ichimoku_crosses(allf_symbol, candles_1d, '1D', False)
            
    symbols = [ "GOG_USDT", "AXL_USDT" ]
    for symbol in symbols:
        candles_4h = get_candles(symbol,"futures",interval="4H",limit=601)
        candles_12h = get_12h_candles_from_4h(candles_4h)
        candles_1d = get_candles(symbol,"futures",interval="1D")
        candles_1W = get_candles(symbol,"futures",interval='Week1')
        candles_1M = get_candles(symbol,"futures",interval='Month1')

        if len(candles_4h) < 14:
            continue 
                
        closes_4h = [float(c[4]) for c in candles_4h]
        closes_12h = [float(c[4]) for c in candles_12h]
        closes_1d = [float(c[4]) for c in candles_1d]
        closes_1W = [float(c[4]) for c in candles_1W]
        closes_1M = [float(c[4]) for c in candles_1M]

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
                
        if change_pct_4h > PRICE_CHANGE_THRESHOLD and rsi_4h and rsi_4h > RSI_THRESHOLD:
            message = f"🚨 {sym_fut}\n4h:{change_pct_4h:.2f}% rsi:{rsi_4h:.2f} macd:{macd_4h_condition}\n1D:{change_pct_1d:.2f}% rsi:{rsi_1d:.2f} macd:{macd_1d_condition}\n W:{change_pct_1W:.2f}% M:{change_pct_1M:.2f}%\n"
            #send_telegram_alert(message)
            print(f"{message}")

if __name__ == "__main__":
    main()
