import requests
#import time
from datetime import datetime
import numpy as np
import pandas as pd

# === PLACEHOLDERS ===
TELEGRAM_TOKEN = "7716430771:AAHqCZNoDACm3qlaue4G_hTJkyrxDRV9uxo"
TELEGRAM_CHAT_ID = "6487259893"

# --- Constants ---
PRICE_CHANGE_THRESHOLD = 10 # in percent
RSI_THRESHOLD = 70
RSI_PERIOD = 14
EMA_LONG_PERIOD = 200
EMA_TOUCH_TOLERANCE = 0.1  # 5% tolerance

def get_perpetual_symbols():
    url = "https://contract.mexc.com/api/v1/contract/detail"
    res = requests.get(url).json()
    return [s["symbol"] for s in res["data"] if s["quoteCoin"] == "USDT"]

def get_candles(symbol, interval='Hour4', limit= EMA_LONG_PERIOD + 1):
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

def calculate_stoch_rsi(closes, rsi_period=14, stoch_period=14, smooth_k=3, smooth_d=3):
    if len(closes) < rsi_period + stoch_period:
        return None, None
    rsi_series = pd.Series([calculate_rsi(closes[i - rsi_period:i + 1], rsi_period) for i in range(rsi_period, len(closes))])
    stoch_rsi = (rsi_series - rsi_series.rolling(stoch_period).min()) / (rsi_series.rolling(stoch_period).max() - rsi_series.rolling(stoch_period).min())
    k = stoch_rsi.rolling(smooth_k).mean()
    d = k.rolling(smooth_d).mean()
    return k.iloc[-1] * 100, d.iloc[-1] * 100
    
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
    #symbols = [ "BTC_USDT", "ETH_USDT", "SOL_USDT", "SUI_USDT", "DOGE_USDT" ]
    symbols = get_perpetual_symbols()
    for symbol in symbols:
        candles_1h = get_candles(symbol, interval='Min60')
        candles_4h = get_candles(symbol,interval='Hour4',limit=(EMA_LONG_PERIOD * 3))
        #candles_12h = get_candles(symbol, interval='Hour12')
        candles_1d = get_candles(symbol,interval='Day1')
        candles_1W = get_candles(symbol,interval='Week1')
        candles_1M = get_candles(symbol,interval='Month1')

        if len(candles_4h) < 14 or len(candles_1d) < 2:
            continue 
        
        closes_1h = [float(c[4]) for c in candles_4h]
        closes_4h = [float(c[4]) for c in candles_4h]
        closes_12h = [
            np.mean(closes_4h[i:i+3]) for i in range(0, len(closes_4h) - 2, 3)
        ]
        closes_1d = [float(c[4]) for c in candles_1d]
        closes_1W = [float(c[4]) for c in candles_1W]
        closes_1M = [float(c[4]) for c in candles_1M]

        change_pct_4h = ((closes_4h[-1] - closes_4h[-2]) / closes_4h[-2]) * 100
        change_pct_1d = ((closes_1d[-1] - closes_1d[-2]) / closes_1d[-2]) * 100
        change_pct_1W = 'n/a'
        change_pct_1M = 'n/a'
        if len(closes_1W) > 2: 
            change_pct_1W = ((closes_1W[-1] - closes_1W[-2]) / closes_1W[-2]) * 100
        if len(closes_1M) > 2:
            change_pct_1M = ((closes_1M[-1] - closes_1M[-2]) / closes_1M[-2]) * 100

        rsi_4h = calculate_rsi(closes_4h)
        rsi_1d = calculate_rsi(closes_1d)

        k_stoch, d_stoch = calculate_stoch_rsi(closes_4h)
        

        macd_4h_condition = False
        macd_1d_condition = False
        if len(candles_1d) >= 50:
            macd_4h, signal_4h, hist_4h = calculate_macd(closes_4h)
            macd_1d, signal_1d, hist_1d = calculate_macd(closes_1d)

            macd_4h_condition = macd_4h > 0 and signal_4h > 0 and hist_4h > 0
            macd_1d_condition = macd_1d > 0 and signal_1d > 0 and hist_1d > 0
            #print(f"{symbol} MACD: {macd_1d:.3f}, Signal: {signal_1d:.3f}, Histogram: {hist_1d:.3f}")

        near_ema_200 = False
        ema_200_12h = -100
        ema_200_1d = -100
        if len(candles_1d) >= EMA_LONG_PERIOD:
            ema_200_12h = calculate_ema(closes_12h)
            ema_200_1d = calculate_ema(closes_1d)
            price = closes_1h[-1]
            #print(f"ema20012h:{ema_200_12h},ema2001d:{ema_200_1d}")

            near_ema_200 = (
                abs(price - ema_200_12h) / ema_200_12h < EMA_TOUCH_TOLERANCE or
                abs(price - ema_200_1d) / ema_200_1d < EMA_TOUCH_TOLERANCE
            )

        # Alert only at 4-hour or 12-hour intervals (UTC)
        now_utc = datetime.utcnow()
        hour = now_utc.hour

        #if hour % 4 == 0 or hour % 12 == 0:
        
        if change_pct_4h > PRICE_CHANGE_THRESHOLD and rsi_4h and rsi_4h > RSI_THRESHOLD:
            message = f"🚨 {symbol}\n4h:{change_pct_4h:.2f}% rsi:{rsi_4h:.2f} macd:{macd_4h_condition}\n1D:{change_pct_1d:.2f}% rsi:{rsi_1d:.2f} macd:{macd_1d_condition}\nema200:{near_ema_200} W:{change_pct_1W:.2f}% M:{change_pct_1M:.2f}%\n"
            send_telegram_alert(message)
            #print(f"{symbol}:{price:.4f} ema20012h:{ema_200_12h:.4f},ema2001d:{ema_200_1d:.4f}")

if __name__ == "__main__":
    main()
