import requests
import time
import hmac
import hashlib
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import csv
import os

Watchlist_Shorts = "watchlists/Shorts.txt"
Watchlist_Longs = "watchlists/Longs.txt"

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

def load_config():
    CONFIG_URL = "https://raw.githubusercontent.com/renaranj/alert-bot-scripts/refs/heads/main/custom_config.txt"
    try:
        r = requests.get(CONFIG_URL)
        r.raise_for_status()
        env = {}
        exec(r.text, {}, env)
        executions = env.get("executions", [])
        return executions
    except Exception as e:
        print("Error loading config:", e)
        return []

def get_allpairs_symbols(market_type = "spot"):
    if market_type == "futures":
        url = "https://contract.mexc.com/api/v1/contract/detail"
        res = requests.get(url).json()
        return [s["symbol"] for s in res["data"] if s["quoteCoin"] == "USDT"]
    elif market_type == "spot":
        url = "https://api.mexc.com/api/v3/exchangeInfo"
        res = requests.get(url).json()
        symbols = [s["symbol"] for s in res["symbols"] if s["quoteAsset"] == "USDT" and s["status"] == "1" and s["isSpotTradingAllowed"]]
        return symbols
    else:
        print(f"Ungultiges Market Type: Getting all pairs symbols failed...")
        
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
                if symbol.startswith("MEXC:"):
                   symbol = symbol.replace("MEXC:", "")
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

def get_candles(symbol, interval, limit= EMA_LONG_PERIOD + 1):
 if symbol.endswith("_USDT"):
    futures_interval_map = {
        "15m": "Min15",
        "30m": "Min30",
        "1h": "Hour1",
        "4h": "Hour4",
        "1d": "Day1",
        "1W": "Week1",
        "1M": "Month1",
    }
    interval = futures_interval_map.get(interval)
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
        
def get_12h_candles_from_4h(candles_4h):
    # Remove the last candle (potentially incomplete)
    closed_candles = candles_4h[:-1]
    if len(closed_candles) < 3:
        return []

    candles_12h = []
    i = 0

    while i <= len(closed_candles) - 3:
        group = closed_candles[i:i + 3]
        ts0 = int(group[0][0])
        dt0 = datetime.utcfromtimestamp(ts0 // 1000 if ts0 > 9999999999 else ts0)

        expected_hours = {
            0: [0, 4, 8],
            12: [12, 16, 20]
        }

        group_hours = [datetime.utcfromtimestamp(int(c[0]) // 1000 if int(c[0]) > 9999999999 else int(c[0])).hour for c in group]

        if group_hours == expected_hours.get(dt0.hour, []):
            o = float(group[0][1])
            h = max(float(c[2]) for c in group)
            l = min(float(c[3]) for c in group)
            c_ = float(group[2][4])
            v = sum(float(c[5]) for c in group)
            candles_12h.append((ts0, o, h, l, c_, v))
            i += 3  # Skip the 3 used
        else:
            i += 1  # Try next alignment

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
    if len(candles) < 200:
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

def alarm_price_crosses(symbol,candles,price, priority=False, debug=False):
    if len(candles) < 2:
        return

    t, o, h, l, c, v = candles[-2]
    h, l = float(h), float(l)

    if l < float(price) < h:
        message = f"crossed price:{price:.4f}\n"
        if debug:
            print(f"{symbol} {message} - (h{h:.4f},l{l:.4f})\n")
        send_telegram_alert(symbol, message, priority)
     
def alarm_price_change(symbol, candles, change_threshold=10, priority=False, debug=False):
    if len(candles) < 3:
        return

    closes = [float(c[4]) for c in candles]
    change_pct = ((closes[-2] - closes[-3]) / closes[-3]) * 100

    if (change_threshold > 0 and change_pct >= change_threshold) or (change_threshold < 0 and change_pct <= change_threshold):
        if "_" in symbol:
            symbol = symbol.replace("_USDT", "USDT.P")
        message = f"Price Changed: {change_pct:.2f}%"
        if debug:
            print(f"{symbol} Price Changed: {change_pct:.2f}% - c1:{closes[-2]:.4f}, c2:{closes[-3]:.4f} \n")
        send_telegram_alert(symbol, message, priority)
    
def alarm_ema200_crosses(symbol, candles_4h, candles_12h, candles_1d, priority=False, debug=False):
    def is_ema_in_candle_range(ema, high, low):
        return low < ema < high

    messages = []

    # 🔹 Check 12H EMA200 against current 4H candle
    if len(candles_12h) >= 200:
        closes_12h = [float(c[4]) for c in candles_12h]
        ema_12h = calculate_ema(closes_12h)
        curr_high, curr_low = float(candles_4h[-1][2]), float(candles_4h[-1][3])

        if is_ema_in_candle_range(ema_12h, curr_high, curr_low):
            messages.append("📌 Touched EMA200 on 12H")

        if debug:
            print(f"{symbol} | 12H EMA: {ema_12h:.4f}, 4H candle: H={curr_high}, L={curr_low}")

    # 🔹 Check 1D EMA200 against current 4H candle
    if len(candles_1d) >= 201:
        closes_1d = [float(c[4]) for c in candles_1d[:-1]]
        ema_1d = calculate_ema(closes_1d)

        if is_ema_in_candle_range(ema_1d, curr_high, curr_low):
            messages.append("📌 Touched EMA200 on 1D")

        if debug:
            print(f"{symbol} | 1D EMA: {ema_1d:.4f}, 4H candle: H={curr_high}, L={curr_low}")

    # 🔹 New: Check if 4H EMA200 is inside full *previous* 4H candle
    if len(candles_4h) >= 200:
        closes_4h = [float(c[4]) for c in candles_4h[:-1]]
        ema_4h = calculate_ema(closes_4h)
        prev_high, prev_low = float(candles_4h[-2][2]), float(candles_4h[-2][3])

        if is_ema_in_candle_range(ema_4h, prev_high, prev_low):
            messages.append("📌 4H EMA200 inside previous full 4H candle")

        if debug:
            print(f"{symbol} | 4H EMA: {ema_4h:.4f}, Prev 4H candle: H={prev_high}, L={prev_low}")

    # 🔔 Send alert if any
    if messages:
        full_msg = f"🔔EMA Signals:\n" + "\n".join(messages)
        send_telegram_alert(symbol, full_msg, priority)

def alarm_candle_patterns(symbol, candles, pattern_name, priority=False, debug=False):
    candles = candles[:-1] if pattern_name != "12H" else candles
    if len(candles) < 3:
        return
    messages = []
    
    # We'll use the last two candles to detect engulfing patterns
    prev = candles[-2]
    curr = candles[-1]

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

    if total_range == 0:
        return "\n".join(messages)

    body_ratio = body / total_range
    upper_ratio = upper_wick / total_range
    lower_ratio = lower_wick / total_range
    
    # Hammer
    #if lower_ratio > 0.6 and upper_ratio < 0.2 and body_ratio < 0.3:
    if lower_ratio > 0.5 and body_ratio < 0.3:
        messages.append(f"🔨 Hammer detected on {pattern_name}")
    # Inverted Hammer
    #elif upper_ratio > 0.6 and lower_ratio < 0.2 and body_ratio < 0.3:
    elif upper_ratio > 0.5 and body_ratio < 0.3:
        messages.append(f"🔻 Inverted Hammer on {pattern_name}")
    # Spinning Top
    elif body_ratio < 0.3 and upper_ratio > 0.3 and lower_ratio > 0.3:
        messages.append(f"🌀 Spinning Top on {pattern_name}")
    
    if messages:
       if debug:
          print(f"{symbol} (o {o:.4f}, h{h:.4f},l{l:.4f},c{c:.4f}) - (bd:{body_ratio:.2f},upp:{upper_ratio:.2f},low:{lower_ratio:.2f})")
       "\n".join(messages)
       send_telegram_alert(symbol, messages, priority)
 
def alarm_ichimoku_crosses(symbol, candles, tf_label="", priority=False, debug=False):
    if len(candles) < 201:
        return ""
    
    # Use latest candle for 12H, otherwise exclude it
    candles = candles if tf_label == "12H" else candles[:-1]
    closes = [float(c[4]) for c in candles]
    prev_close = closes[-2]
    curr_close = closes[-1]

    messages = []

    # Calculate Ichimoku components
    tenkan, kijun, senkou_a, senkou_b = calculate_ichimoku(candles)

    # Ensure Senkou spans are available
    if len(senkou_a) < 27 or len(senkou_b) < 27:
        return ""

    prev_senkou_a = senkou_a.iloc[-2]
    prev_senkou_b = senkou_b.iloc[-2]
    curr_senkou_a = senkou_a.iloc[-1]
    curr_senkou_b = senkou_b.iloc[-1]

    prev_top = max(prev_senkou_a, prev_senkou_b)
    prev_bottom = min(prev_senkou_a, prev_senkou_b)
    curr_top = max(curr_senkou_a, curr_senkou_b)
    curr_bottom = min(curr_senkou_a, curr_senkou_b)

    # ✅ Condition 1: Previous inside cloud, current outside
    if prev_bottom <= prev_close <= prev_top and (curr_close < curr_bottom or curr_close > curr_top):
        messages.append(f"📤 Price exited Ichimoku cloud on {tf_label}")

    # ✅ Condition 2: Flip from above to below or vice versa
    if (prev_close > prev_top and curr_close < curr_bottom) or (prev_close < prev_bottom and curr_close > curr_top):
        messages.append(f"🔁 Price flipped sides across the cloud on {tf_label}")

    # Tenkan/Kijun Cross
    if tenkan.iloc[-2] < kijun.iloc[-2] and tenkan.iloc[-1] >= kijun.iloc[-1]:
        if curr_close > curr_top:
            messages.append(f"🟢 Bullish Tenkan/Kijun cross above cloud on {tf_label}")
        else:
            messages.append(f"🟡 Bullish Tenkan/Kijun cross below/inside cloud on {tf_label}")
    elif tenkan.iloc[-2] > kijun.iloc[-2] and tenkan.iloc[-1] <= kijun.iloc[-1]:
        if curr_close < curr_bottom:
            messages.append(f"🔴 Bearish Tenkan/Kijun cross below cloud on {tf_label}")
        else:
            messages.append(f"🟠 Bearish Tenkan/Kijun cross above/inside cloud on {tf_label}")

    if messages:
        combined_msg = "\n".join(messages)
        if debug:
            print(f"[{symbol}]\n{messages}")
            print(f"Ichimoku Cloud previous {prev_close:.4f}:({prev_top:.4f},{prev_bottom:.4f}), current {curr_close:.4f}: ({curr_top:.4f},{curr_bottom:.4f})")
            print(f"Tenkan/Kijun previous:({tenkan.iloc[-2]:.4f},{kijun.iloc[-2]:.4f}), current:({tenkan.iloc[-1]:.4f},{kijun.iloc[-1]:.4f})")
        send_telegram_alert(symbol, combined_msg, priority)
                       
def send_telegram_alert(symbol, message, priority):
    if "_" in symbol:
       symbol = symbol.replace("_USDT", "USDT.P")
    prefix = "🚨🚨" if priority else ""
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

        if hour in [0,4,8,12,16,20] and now.minute in [0,1,2,3,4,5]:
           
           #-----------BTCUSDT bearbeitung---------------------------------------------------#
           candles_4h = get_candles("BTCUSDT","4h",limit=601)
           candles_12h = get_12h_candles_from_4h(candles_4h)
           candles_1d = get_candles("BTCUSDT","1d")
           alarm_ema200_crosses("BTCUSDT", candles_4h, candles_12h, candles_1d, True)
           alarm_candle_patterns("BTCUSDT", candles_4h, "4H", True)
           if hour in [0,12]:
             alarm_candle_patterns("BTCUSDT", candles_12h, "12H", True)
           if hour in [0]:
             alarm_candle_patterns("BTCUSDT", candles_1d, "1D", True)
           
            #bearbeitung meine Coins
           send_telegram_alert("MX_USDT", "Bearbeitung meine Spot Werts",True)
           symbols = get_open_symbols("spot")
           #open_spots = []    
           for symbol in symbols:
               candles_4h = get_candles(symbol,"4h",limit=601)
               candles_12h = get_12h_candles_from_4h(candles_4h)
               candles_1d = get_candles(symbol,"1d")
               alarm_price_change(symbol, candles_4h, 20, True)
               alarm_candle_patterns(symbol, candles_4h, "4H", True)
               if hour in [0,12]:
                    alarm_candle_patterns(symbol, candles_12h, "12H", True)
               if hour in [0]:
                    alarm_candle_patterns(symbol, candles_1d, "1D", True)   
           
            #Coins wo ich position behalte            
           symbols = get_open_symbols("futures")
           #open_futures = []
           for symbol in symbols:
               candles_4h = get_candles(symbol,"4h",limit=601)
               candles_12h = get_12h_candles_from_4h(candles_4h)
               candles_1d = get_candles(symbol,"1d")
               alarm_price_change(symbol, candles_4h, -20, True)
               alarm_candle_patterns(symbol, candles_4h, "4H", True)
               if hour in [0,12]:
                    alarm_candle_patterns(symbol, candles_12h, "12H", True)
               if hour in [0]:
                    alarm_candle_patterns(symbol, candles_1d, "1D", True)
           
            #Beobachtung my persönliches List coins
           send_telegram_alert("MX_USDT", "Bearbeitung meine Watchlists",True)
           symbols = load_watchlist_from_csv(Watchlist_Shorts)
           for symbol in symbols:
               candles_4h = get_candles(symbol,"4h",limit=601)
               candles_12h = get_12h_candles_from_4h(candles_4h)
               candles_1d = get_candles(symbol,"1d")
               closes_4h = [float(c[4]) for c in candles_4h]
               alarm_ichimoku_crosses(symbol, candles_4h, '4H')
               stoch_rsiK, stoch_rsiD = calculate_stoch_rsi(closes_4h)
               if stoch_rsiK and stoch_rsiD and (stoch_rsiK < 20 or stoch_rsiK > 80) and (stoch_rsiD < 20 or stoch_rsiD > 80):
                  if hour in [0,12]:
                    alarm_candle_patterns(symbol, candles_12h, "12H")
                  if hour in [0]:
                    alarm_candle_patterns(symbol, candles_1d, "1D")   
               
           #Beobachtung all pair futures
           send_telegram_alert("MX_USDT", "Bearbeitung all pairs",True)
           symbols = get_allpairs_symbols("futures")
           for symbol in symbols:
               candles_4h = get_candles(symbol,"4h",limit=601)
               candles_12h = get_12h_candles_from_4h(candles_4h)
               candles_1d = get_candles(symbol,"1d")
               if hour in [0,12]:
                   alarm_ichimoku_crosses(symbol, candles_12h, '12H') 
               if hour in [0]:
                   alarm_ichimoku_crosses(symbol, candles_1d, '1D')
               alarm_price_change(symbol, candles_4h, 10, True)
               alarm_ema200_crosses(symbol, candles_4h, candles_12h, candles_1d, True)
            
        else:
             symbols = ["HYPE_USDT","XPR_USDT","FARTCOIN_USDT","OBOL_USDT"]
             #symbols = []
             for symbol in symbols:
                 candles_4h = get_candles(symbol,"4h",limit=601)
                 candles_12h = get_12h_candles_from_4h(candles_4h)
                 candles_1d = get_candles(symbol,"1d")
                 alarm_ema200_crosses(symbol, candles_4h, candles_12h, candles_1d, True,True)
                 alarm_ichimoku_crosses(symbol, candles_4h, '4H',False,False)
             return 
            
             print(f"executing load config...")
             executions = load_config()
             if not executions:
                 print("No execution rules found.")
                 return

             FUNC_MAP = {
                    "price_change": alarm_price_change,
                    "candle_patterns": alarm_candle_patterns,
                    "ema200_crosses": alarm_ema200_crosses,
                    "ichimoku_crosses": alarm_ichimoku_crosses,
                    "price_crosses": alarm_price_crosses,
                    # Add more functions as needed
             }

             for symbol, functions_dict in executions:
                 for func_name, input in functions_dict.items():
                     func = FUNC_MAP.get(func_name)
                     if not func:
                         print(f"[WARN] Unknown function: {func_name}")
                         continue
  
                     # Dispatch with appropriate arguments
                     if func_name == "candle_patterns":
                         candles = get_candles(symbol, input, limit=601)
                         func(symbol, candles, input, True)
                         print(f"{func} -> {input}")
                     elif func_name == "price_change":
                         candles = get_candles(symbol, "15m", limit=601)
                         func(symbol, candles, float(input), True)
                     elif func_name == "ema200_crosses":
                         candles_15m = get_candles(symbol,"15m",limit=3)
                         candles_4h = get_candles(symbol,"4h",limit=601)
                         candles_12h = get_12h_candles_from_4h(candles_4h)
                         candles_1d = get_candles(symbol,"1d")
                         func(symbol, candles_15m, candles_12h, candles_1d, True)
                     elif func_name == "ichimoku_crosses":
                         candles = get_candles(symbol, input, limit=601)
                         func(symbol, candles, input, True)
                     elif func_name == "price_crosses":
                         candles = get_candles(symbol, "15min", limit=601)
                         func(symbol, candles, float(input), True)
                     else:
                         print(f"[WARN] No logic implemented for: {func_name}")
        
if __name__ == "__main__":
    main()
