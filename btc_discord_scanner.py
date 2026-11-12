import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from numba import njit
from binance.client import Client
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, TakeProfitRequest, StopLossRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

# ==========================================
# 1. CONFIGURATION & SECRETS CHECK
# ==========================================
ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")

if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
    raise ValueError("❌ CRITICAL ERROR: Alpaca API Key or Secret Key is missing from GitHub Secrets!")

binance_client = Client(tld='us') 
trading_client = TradingClient(api_key=ALPACA_API_KEY, secret_key=ALPACA_SECRET_KEY, paper=True)

TICKER = "BTCUSDT"
ALPACA_SYMBOL = "BTC/USD"
RISK_PCT = 0.01 # 1.0% Risk per trade 

# ==========================================
# 2. DISCORD ALERT MODULES
# ==========================================
def ping_discord_startup():
    payload = {
        "username": "V6 AMT Sniper",
        "embeds": [{
            "title": "🟢 V6 Matrix Engine Online",
            "description": "Cloud runner spun up successfully. Scanning BTCUSD 1m order flow for 70% Value Area sweeps...",
            "color": 3066993, 
            "footer": {"text": "Bot is active for the next 6 hours."}
        }]
    }
    try:
        requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
    except Exception as e:
        print(f"Failed to ping Discord startup: {e}")

def send_discord_alert(action, price, balance, sl, tp, be, vah, val):
    color = 65280 if action == "BUY" else 16711680
    risk_amount = balance * RISK_PCT
    payload = {
        "username": "V6 AMT Sniper",
        "embeds": [{
            "title": f"🚨 BTCUSD | Failed Auction Confirmed",
            "color": color,
            "fields": [
                {"name": "Action", "value": f"**{action}** @ ${price:.2f}", "inline": True},
                {"name": "Risk (1.0%)", "value": f"${risk_amount:,.2f}", "inline": True},
                {"name": "70% Value Area", "value": f"VAH: ${vah:.2f} | VAL: ${val:.2f}", "inline": False},
                {"name": "Trade Management", "value": f"**🎯 TP (1:2):** ${tp:.2f}\n**🛡️ BE Trigger (1:1):** ${be:.2f}\n**🛑 SL (Initial):** ${sl:.2f}", "inline": False}
            ],
            "footer": {"text": "V6 Matrix Engine: Vol Spike + Macro Trend Active ⚡"}
        }]
    }
    requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)

# ==========================================
# 3. NUMBA V6 AMT ENGINE
# ==========================================
@njit
def fast_amt_scan(closes, highs, lows, volumes, cvds, atrs, vol_mas, emas_1440):
    if len(closes) < 1440:
        return 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        
    c_close = closes[-1]
    c_high = highs[-1]
    c_low = lows[-1]
    c_atr = atrs[-1]

    window_closes = closes[-480:]
    window_vols = volumes[-480:]
    min_c = np.min(window_closes)
    max_c = np.max(window_closes)
    
    bin_size = (max_c - min_c) / 20.0
    binned_vols = np.zeros(20)
    for j in range(480):
        idx = int((window_closes[j] - min_c) / bin_size)
        if idx >= 20: idx = 19
        binned_vols[idx] += window_vols[j]
        
    total_vol = np.sum(binned_vols)
    poc_idx = np.argmax(binned_vols)
    
    va_vol = binned_vols[poc_idx]
    target_vol = total_vol * 0.70
    upper_idx = lower_idx = poc_idx
    
    while va_vol < target_vol:
        vol_up = binned_vols[upper_idx + 1] if upper_idx < 19 else -1.0
        vol_down = binned_vols[lower_idx - 1] if lower_idx > 0 else -1.0
        if vol_up == -1.0 and vol_down == -1.0: break
        if vol_up >= vol_down:
            upper_idx += 1; va_vol += vol_up
        else:
            lower_idx -= 1; va_vol += vol_down
            
    vah = min_c + (upper_idx * bin_size) + (bin_size / 2.0)
    val = min_c + (lower_idx * bin_size) + (bin_size / 2.0)

    lowest_20 = np.min(lows[-20:-1])
    highest_20 = np.max(highs[-20:-1])
    
    failed_auction_low = (lowest_20 < val) and (c_low <= lowest_20) and (c_close > val)
    failed_auction_high = (highest_20 > vah) and (c_high >= highest_20) and (c_close < vah)

    trend_bull = c_close > emas_1440[-1]
    trend_bear = c_close < emas_1440[-1]
    vol_ok = volumes[-1] > (1.5 * vol_mas[-2])

    cvd_bull_div = cvds[-1] > cvds[-2]
    cvd_bear_div = cvds[-1] < cvds[-2]

    if failed_auction_low and cvd_bull_div and vol_ok and trend_bull:
        return 1, c_close, c_close - (c_atr*2), c_close + (c_atr*4), c_close + (c_atr*2), vah, val
    elif failed_auction_high and cvd_bear_div and vol_ok and trend_bear:
        return -1, c_close, c_close + (c_atr*2), c_close - (c_atr*4), c_close - (c_atr*2), vah, val

    return 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

def generate_signals():
    try:
        klines = binance_client.get_klines(symbol=TICKER, interval=Client.KLINE_INTERVAL_1MINUTE, limit=1500)
        df = pd.DataFrame(klines, columns=['time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'])
        
        for col in ['open', 'high', 'low', 'close', 'volume', 'taker_base_vol']:
            df[col] = df[col].astype(float)
            
        df['market_sell'] = df['volume'] - df['taker_base_vol']
        df['delta'] = df['taker_base_vol'] - df['market_sell']
        df['cvd'] = df['delta'].cumsum()
        
        df['tr'] = np.maximum(df['high'] - df['low'], 
                              np.maximum(abs(df['high'] - df['close'].shift(1)), 
                                         abs(df['low'] - df['close'].shift(1))))
        df['atr_20'] = df['tr'].rolling(20).mean().bfill()
        df['vol_ma_20'] = df['volume'].rolling(20).mean().bfill()
        df['ema_1440'] = df['close'].ewm(span=1440, adjust=False).mean()
        
        closes = df['close'].to_numpy(dtype=np.float64)
        highs = df['high'].to_numpy(dtype=np.float64)
        lows = df['low'].to_numpy(dtype=np.float64)
        volumes = df['volume'].to_numpy(dtype=np.float64)
        cvds = df['cvd'].to_numpy(dtype=np.float64)
        atrs = df['atr_20'].to_numpy(dtype=np.float64)
        vol_mas = df['vol_ma_20'].to_numpy(dtype=np.float64)
        emas_1440 = df['ema_1440'].to_numpy(dtype=np.float64)

        return fast_amt_scan(closes, highs, lows, volumes, cvds, atrs, vol_mas, emas_1440)
        
    except Exception as e:
        print(f"Data error: {e}")
        return 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

# ==========================================
# 4. MAIN RUNNER (6-Hour GitHub Actions Loop)
# ==========================================
if __name__ == "__main__":
    print("V6 AMT Sniper Session Initiated...")
    
    # 🟢 Send startup ping to Discord
    ping_discord_startup()
    
    end_time = time.time() + (5.9 * 3600) 
    
    while time.time() < end_time:
        try:
            signal, price, sl, tp, be, vah, val = generate_signals()
            
            if signal != 0:
                side = "BUY" if signal == 1 else "SELL"
                print(f"✅ V6 Failed Auction! {side} @ ${price:.2f}")
                
                account = trading_client.get_account()
                balance = float(account.equity)
                
                send_discord_alert(side, price, balance, sl, tp, be, vah, val)
                
                if signal == 1:
                    qty = round((balance * RISK_PCT) / abs(price - sl), 4)
                    bracket_order = MarketOrderRequest(
                        symbol=ALPACA_SYMBOL,
                        qty=qty,
                        side=OrderSide.BUY,
                        time_in_force=TimeInForce.GTC,
                        order_class=OrderClass.BRACKET,
                        take_profit=TakeProfitRequest(limit_price=round(tp, 2)),
                        stop_loss=StopLossRequest(stop_price=round(sl, 2))
                    )
                    trading_client.submit_order(bracket_order)
                    print("Long executed on Alpaca Paper.")
                else:
                    print("Short signal routed to Discord. (Alpaca API blocks crypto shorts).")
                    
                time.sleep(60)
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ⏳ Scanning 70% Value Area for sweeps...")
                time.sleep(60)
                
        except Exception as e:
            print(f"Loop error: {e}. Retrying in 60s...")
            time.sleep(60)
