import os
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone
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
    raise ValueError("CRITICAL ERROR: ALPACA_API_KEY or ALPACA_SECRET_KEY is missing from GitHub Secrets!")

if not DISCORD_WEBHOOK_URL:
    raise ValueError("CRITICAL ERROR: DISCORD_WEBHOOK_URL is missing from GitHub Secrets!")

binance_client = Client(tld='us')
trading_client = TradingClient(api_key=ALPACA_API_KEY, secret_key=ALPACA_SECRET_KEY, paper=True)

TICKER = "BTCUSDT"
ALPACA_SYMBOL = "BTC/USD"
RISK_PCT = 0.01          # 1.0% Risk per trade ($50 on $5k Prop / $1,000 on $100k Alpaca Paper)
LATCH_COOLDOWN_SEC = 360 # 6-minute cooldown so a 6-bar latch never double-fires on the same sweep

# ==========================================
# 2. DISCORD ALERT MODULES
# ==========================================
def ping_discord_startup():
    payload = {
        "username": "V6.2 AMT Sniper",
        "embeds": [{
            "title": "🟢 V6.2 Matrix Engine Online",
            "description": (
                "Cloud runner spun up successfully on GitHub Actions.\n"
                "**Active Upgrade:** 6-Bar (6m) Sweep Memory Latch + Fast 21 EMA Reclaim + 1440 EMA Macro Filter.\n"
                "Scanning BTCUSD 1m order flow for 70% Value Area sweeps..."
            ),
            "color": 3066993,
            "footer": {"text": "Bot is active for the next 5h 54m cycle | 1.0% Risk Bracket Locked"}
        }]
    }
    try:
        resp = requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
        print(f"Startup ping sent to Discord (HTTP {resp.status_code})")
    except Exception as e:
        print(f"Failed to ping Discord startup: {e}")

def send_discord_alert(action, price, balance, sl, tp, be, vah, val, sweep_age):
    color = 65280 if action == "BUY" else 16711680
    risk_amount = balance * RISK_PCT
    exec_note = "Auto-Executed on Alpaca Paper" if action == "BUY" else "Discord Manual Short (Spot Short Blocked)"
    payload = {
        "username": "V6.2 AMT Sniper",
        "embeds": [{
            "title": f"🚨 BTCUSD V6.2 | {action} Failed Auction Confirmed",
            "color": color,
            "fields": [
                {"name": "Action", "value": f"**{action}** @ `${price:,.2f}`", "inline": True},
                {"name": "Risk (1.0%)", "value": f"`${risk_amount:,.2f}`", "inline": True},
                {"name": "Sweep Memory Latch", "value": f"Swept `{sweep_age}m` ago | {exec_note}", "inline": False},
                {"name": "8H 70% Value Area", "value": f"VAH: `${vah:,.2f}` | VAL: `${val:,.2f}`", "inline": False},
                {
                    "name": "Trade Management (1:2 Bracket)",
                    "value": (
                        f"🎯 **TP (1:2 / 4x ATR):** `${tp:,.2f}`\n"
                        f"🛡️ **BE Trigger (1:1 / 2x ATR):** `${be:,.2f}`\n"
                        f"🛑 **SL (Initial / 2x ATR):** `${sl:,.2f}`"
                    ),
                    "inline": False
                }
            ],
            "footer": {"text": f"V6.2 Matrix Engine: 6m Latch + 21/1440 EMA + 1.5x Vol Spike • {datetime.now(timezone.utc).strftime('%H:%M UTC')}"}
        }]
    }
    try:
        requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=10)
    except Exception as e:
        print(f"Failed to send Discord trade alert: {e}")

# ==========================================
# 3. NUMBA V6.2 AMT ENGINE (6-BAR LATCH + 21/1440 EMA)
# ==========================================
@njit
def fast_amt_scan(closes, opens, highs, lows, volumes, cvds, atrs, vol_mas, emas_21, emas_1440):
    if len(closes) < 1440:
        return 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0

    c_close = closes[-1]
    c_open  = opens[-1]
    c_atr   = atrs[-1]

    # 1. 8-Hour (480-Bar) 70% Value Area Profile
    window_closes = closes[-480:]
    window_vols   = volumes[-480:]
    min_c = np.min(window_closes)
    max_c = np.max(window_closes)

    bin_size = (max_c - min_c) / 20.0
    if bin_size <= 0.0:
        return 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0

    binned_vols = np.zeros(20)
    for j in range(480):
        idx = int((window_closes[j] - min_c) / bin_size)
        if idx >= 20:
            idx = 19
        binned_vols[idx] += window_vols[j]

    total_vol = np.sum(binned_vols)
    poc_idx = np.argmax(binned_vols)

    va_vol = binned_vols[poc_idx]
    target_vol = total_vol * 0.70
    upper_idx = lower_idx = poc_idx

    while va_vol < target_vol:
        vol_up = binned_vols[upper_idx + 1] if upper_idx < 19 else -1.0
        vol_down = binned_vols[lower_idx - 1] if lower_idx > 0 else -1.0
        if vol_up == -1.0 and vol_down == -1.0:
            break
        if vol_up >= vol_down:
            upper_idx += 1
            va_vol += vol_up
        else:
            lower_idx -= 1
            va_vol += vol_down

    vah = min_c + (upper_idx * bin_size) + (bin_size / 2.0)
    val = min_c + (lower_idx * bin_size) + (bin_size / 2.0)

    # 2. 6-Bar (6-Minute) Sweep Memory Latch & 1.5x Volume Spike Memory
    bull_sweep_latch = False
    bear_sweep_latch = False
    vol_spike_latch  = False
    bull_sweep_age   = 0
    bear_sweep_age   = 0

    for k in range(1, 7):  # Inspect last 6 closed 1m candles (-1 to -6)
        idx = -k
        prior_low_20  = np.min(lows[idx - 20:idx])
        prior_high_20 = np.max(highs[idx - 20:idx])

        if (prior_low_20 < val) and (lows[idx] <= prior_low_20):
            if not bull_sweep_latch:
                bull_sweep_age = k - 1
            bull_sweep_latch = True

        if (prior_high_20 > vah) and (highs[idx] >= prior_high_20):
            if not bear_sweep_latch:
                bear_sweep_age = k - 1
            bear_sweep_latch = True

        if volumes[idx] > (1.5 * vol_mas[idx - 1]):
            vol_spike_latch = True

    # 3. Macro 1440 EMA Trend + Fast 21 EMA Reclaim + Value Area Re-Entry
    trend_bull = c_close > emas_1440[-1]
    trend_bear = c_close < emas_1440[-1]

    bull_reclaim = (c_close > val) and (c_close > emas_21[-1]) and (c_close > c_open)
    bear_reclaim = (c_close < vah) and (c_close < emas_21[-1]) and (c_close < c_open)

    # 4. 3-Bar Cumulative Volume Delta (CVD) Slope Confirmation
    cvd_bull_div = cvds[-1] > cvds[-3]
    cvd_bear_div = cvds[-1] < cvds[-3]

    if bull_sweep_latch and vol_spike_latch and bull_reclaim and cvd_bull_div and trend_bull:
        return 1, c_close, c_close - (c_atr * 2.0), c_close + (c_atr * 4.0), c_close + (c_atr * 2.0), vah, val, bull_sweep_age
    elif bear_sweep_latch and vol_spike_latch and bear_reclaim and cvd_bear_div and trend_bear:
        return -1, c_close, c_close + (c_atr * 2.0), c_close - (c_atr * 4.0), c_close - (c_atr * 2.0), vah, val, bear_sweep_age

    return 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0

def generate_signals():
    try:
        klines = binance_client.get_klines(symbol=TICKER, interval=Client.KLINE_INTERVAL_1MINUTE, limit=1500)
        df = pd.DataFrame(klines, columns=[
            'time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol', 'ignore'
        ])

        for col in ['open', 'high', 'low', 'close', 'volume', 'taker_base_vol']:
            df[col] = df[col].astype(float)

        df['market_sell'] = df['volume'] - df['taker_base_vol']
        df['delta'] = df['taker_base_vol'] - df['market_sell']
        df['cvd'] = df['delta'].cumsum()

        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr_20'] = df['tr'].rolling(20).mean().bfill()
        df['vol_ma_20'] = df['volume'].rolling(20).mean().bfill()
        df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
        df['ema_1440'] = df['close'].ewm(span=1440, adjust=False).mean()

        closes    = df['close'].to_numpy(dtype=np.float64)
        opens     = df['open'].to_numpy(dtype=np.float64)
        highs     = df['high'].to_numpy(dtype=np.float64)
        lows      = df['low'].to_numpy(dtype=np.float64)
        volumes   = df['volume'].to_numpy(dtype=np.float64)
        cvds      = df['cvd'].to_numpy(dtype=np.float64)
        atrs      = df['atr_20'].to_numpy(dtype=np.float64)
        vol_mas   = df['vol_ma_20'].to_numpy(dtype=np.float64)
        emas_21   = df['ema_21'].to_numpy(dtype=np.float64)
        emas_1440 = df['ema_1440'].to_numpy(dtype=np.float64)

        return fast_amt_scan(closes, opens, highs, lows, volumes, cvds, atrs, vol_mas, emas_21, emas_1440)

    except Exception as e:
        print(f"Data error: {e}")
        return 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0

# ==========================================
# 4. MAIN RUNNER (6-HOUR GITHUB ACTIONS LOOP)
# ==========================================
if __name__ == "__main__":
    print("V6.2 AMT Sniper Session Initiated...")

    # 1. Send startup heartbeat ping to Discord
    ping_discord_startup()

    end_time = time.time() + (5.9 * 3600)
    last_signal_time = 0.0

    while time.time() < end_time:
        try:
            signal, price, sl, tp, be, vah, val, sweep_age = generate_signals()
            cooldown_ready = (time.time() - last_signal_time) > LATCH_COOLDOWN_SEC

            if signal != 0 and cooldown_ready:
                side = "BUY" if signal == 1 else "SELL"
                print(f"✅ V6.2 Failed Auction! {side} @ ${price:,.2f} (Sweep Age: {sweep_age}m)")

                account = trading_client.get_account()
                balance = float(account.equity)

                send_discord_alert(side, price, balance, sl, tp, be, vah, val, sweep_age)

                if signal == 1:
                    qty = round((balance * RISK_PCT) / abs(price - sl), 4)
                    if qty > 0:
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
                        print(f"Long bracket order ({qty} BTC) executed on Alpaca Paper.")
                else:
                    print("Short signal routed to Discord. (Alpaca API blocks crypto spot shorts).")

                last_signal_time = time.time()
                time.sleep(60)
            else:
                print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}] ⏳ Scanning 70% Value Area (6m Latch + 21/1440 EMA)...")
                time.sleep(60)

        except Exception as e:
            print(f"Loop error: {e}. Retrying in 60s...")
            time.sleep(60)
