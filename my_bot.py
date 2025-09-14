#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ccxt
import time
import math
import json
import os
from datetime import datetime, timezone, date

import pandas as pd
import numpy as np

# =========================
#      Р Е Ж И М Ы
# =========================
MODE = "LIVE"  # "TEST" (скальпинг, частые проверки) или "LIVE" (строгий боевой)

PRESETS = {
    "LIVE": {
        # Торгуем реже, консервативно
        "TIMEFRAME_ENTRY": "5m",              #было 1h
        "TIMEFRAME_HTF": "1h",                #было 4h
        "AVOID_HOURS_UTC": set(range(0, 6)),  # ночи не торгуем
        "MAX_DAILY_DD": 0.05,                 # 5% дневной стоп
        "MAX_ATR_PCT": 0.03,                  # не торгуем при сверхволатильности
        "RSI_OVERSOLD": 30,
        "RSI_OVERBOUGHT": 70,
        "USE_IMPULSE": True,                  # запасной импульсный вход
        "LOOKBACK_BREAKOUT": 20,
        "DRY_RUN": False,                     # в бою торгуем реально
    },
    "TEST": {
        # Скальпинг: чаще проверки, мягкие фильтры, все часы
        "TIMEFRAME_ENTRY": "5m",
        "TIMEFRAME_HTF": "1h",
        "AVOID_HOURS_UTC": set(),             # торгуем 24/7
        "MAX_DAILY_DD": 2.0,                  # отключаем дневной стоп
        "MAX_ATR_PCT": 0.06,                  # терпим больше волатильности
        "RSI_OVERSOLD": 35,                   # легче получить лонг
        "RSI_OVERBOUGHT": 65,                 # легче получить шорт
        "USE_IMPULSE": True,                  # добавим пробой диапазона
        "LOOKBACK_BREAKOUT": 12,              # более короткий диапазон
        "DRY_RUN": True,                      # тестируем безопасно
    },
}

CFG = PRESETS[MODE]

# =========================
#          К О Н Ф И Г
# =========================
BINGX_API_KEY    = os.getenv('BINGX_API_KEY')
BINGX_SECRET_KEY = os.getenv('BINGX_SECRET_KEY')
if not BINGX_API_KEY or not BINGX_SECRET_KEY:
    raise RuntimeError("BINGX_API_KEY / BINGX_SECRET_KEY are not set in environment")

# --- Вселенная монет ---
UNIVERSE_MODE   = "DYNAMIC"   # "DYNAMIC" или "STATIC"
UNIVERSE_TOP_N  = 30          # сколько лучших по объёму перпетуалов брать
UNIVERSE_EXCLUDE = {          # чёрный список (если хочешь что-то исключить)
    # пример: '1000PEPE', 'BTC3L', ...
    'ETH', 'BTC', 'SOL',
}
# Статический fallback (если DYNAMIC выключен или не удалось собрать вселенную)
SYMBOLS_TO_TRADE_STATIC = [
    'BTC/USDT:USDT','ETH/USDT:USDT','BNB/USDT:USDT','SOL/USDT:USDT','XRP/USDT:USDT',
    'LINK/USDT:USDT','DOGE/USDT:USDT','AVAX/USDT:USDT','LTC/USDT:USDT','ADA/USDT:USDT',
    'DOT/USDT:USDT','TRX/USDT:USDT','NEAR/USDT:USDT','ATOM/USDT:USDT','FIL/USDT:USDT',
    'APT/USDT:USDT','ARB/USDT:USDT','OP/USDT:USDT','INJ/USDT:USDT','SUI/USDT:USDT',
    'SEI/USDT:USDT','TIA/USDT:USDT','AAVE/USDT:USDT','UNI/USDT:USDT','ETC/USDT:USDT',
]

# Явно деривативы (swap) на BingX: добавляем :USDT
SYMBOLS_TO_TRADE = [
    'BTC/USDT:USDT', 'ETH/USDT:USDT',
    'BNB/USDT:USDT', 'SOL/USDT:USDT', 'XRP/USDT:USDT',
    'LINK/USDT:USDT', 'POL/USDT:USDT', 'DOGE/USDT:USDT',
    'AVAX/USDT:USDT', 'LTC/USDT:USDT',
]

# Риск/менеджмент
LEVERAGE         = 35
RISK_PER_TRADE   = 0.005     # 0.5% от эквити на сделку
RISK_REWARD      = 1.5
SL_ATR_MULT      = 2.0
TP_ATR_MULT      = SL_ATR_MULT * RISK_REWARD
MAX_NOTIONAL_PCT = 0.01      # не более 1% эквити (номинал, с плечом)
FEE_PCT          = 0.0006
SLIPPAGE_PCT     = 0.0005

# Funding-фильтры (ставка за 8ч)
FUNDING_SOFT_ABS = 0.0005   # 0.05%: мягкое ограничение — режем риск x0.5
FUNDING_HARD_ABS = 0.0010   # 0.10%: жёсткая блокировка входа
FUNDING_NO_OPEN_WINDOW_SEC = 3600  # не открываемся за 60 мин до невыгодного funding

# Портфельные лимиты
MAX_OPEN_POS     = 3

# Кулдаун
TRADE_COOLDOWN_BARS = 3      # поменьше в TEST

# Управление позицией
BREAKEVEN_AT_R        = 0.8
TRAIL_AT_R            = 1.2
TRAIL_ATR_MULT        = 1.8
PARTIAL_TP_ENABLE     = True
PARTIAL_TP_R          = 0.8
PARTIAL_TP_PART       = 0.5
SECOND_TP_ENABLE      = True
SECOND_TP_R           = 1.8   # раньше 2.0

# Индикаторы
RSI_PERIOD = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
ATR_PERIOD = 14
EMA_TREND_PERIOD = 200
EMA_FAST, EMA_SLOW = 20, 50
VWAP_PERIOD = 20
KELTNER_ATR_MULT = 1.5
VOL_LOOKBACK = 30
VOL_ANOM_MULT = 2.0

# Технические
DRY_RUN          = CFG["DRY_RUN"]                         # важный флаг!
LOG_FILE         = '/home/{}/crypto-bot/trades_log.jsonl'.format(os.getenv('USER', 'ubuntu'))
PRINT_PREFIX     = '[BOT]'

TIMEFRAME_ENTRY = CFG["TIMEFRAME_ENTRY"]
TIMEFRAME_HTF   = CFG["TIMEFRAME_HTF"]
AVOID_HOURS_UTC = CFG["AVOID_HOURS_UTC"]
MAX_DAILY_DD    = CFG["MAX_DAILY_DD"]
MAX_ATR_PCT     = CFG["MAX_ATR_PCT"]
RSI_OVERSOLD    = CFG["RSI_OVERSOLD"]
RSI_OVERBOUGHT  = CFG["RSI_OVERBOUGHT"]
USE_IMPULSE     = CFG["USE_IMPULSE"]
LOOKBACK_BREAKOUT = CFG["LOOKBACK_BREAKOUT"]

# Pump Alert / Lock
PUMP_LOCK_BARS = 3
LATE_BREAKOUT_ATR_MULT = 1.2   # «слишком поздний» выход за Keltner без объёма

# --- PrePump Detector ---
PREPUMP_ENABLE            = True
PREPUMP_ENTER             = True       # если False — только помечаем в лог, без входа
PREPUMP_MAX_RISK          = 0.002      # 0.2% equity на пробный вход (меньше стандартных 0.5%)
PREPUMP_LEVERAGE          = 5          # небольшое плечо для пробного входа
PREPUMP_VOL_LOOKBACK      = 6          # анализ последних N баров
PREPUMP_MIN_RISING_BARS   = 3          # минимум подряд растущих объёмов
PREPUMP_MAX_FLAT_PCT      = 0.006      # цена «флэт»: (max-min)/close < 0.6%
PREPUMP_BOOK_TOP_LEVELS   = 10         # сколько уровней стакана суммируем
PREPUMP_BID_ASK_RATIO_MIN = 2.0        # дисбаланс стакана для LONG: bids >= 2x asks
PREPUMP_OI_MIN_GROWTH_PCT = 0.05       # +5% OI за бар (если доступно из API — опционально)

# --- Жёсткое обеспечение TP/SL и авто-сопровождение ---
ENFORCE_TP_SL          = True     # перепроверять и восстанавливать TP/SL на каждом цикле
TP_AS_PARTIAL          = True     # TP частичный (доля от позиции), иначе 100%
TP_PART_FRACTION       = PARTIAL_TP_PART if PARTIAL_TP_ENABLE else 1.0  # доля позиции на TP
SL_EDIT_VIA_EDITORDER  = False    # если биржа поддерживает editOrder() — используем
TP_REPLACE_IF_MISSED   = True     # если TP не найден среди открытых ордеров — поставить заново
SL_REPLACE_IF_MISSED   = True     # если SL не найден — поставить заново
TP_PRICE_PADDING_TICKS = 0        # 0 = точная цена; >0 — смещение на n тиков (иногда нужно для исполнения)
SL_MARKET_PARAMS_LIST  = [
    {'type': 'stop', 'stopPrice': None, 'reduceOnly': True},
    {'type': 'stop', 'triggerPrice': None, 'reduceOnly': True},
    {'stopLossPrice': None, 'reduceOnly': True},
]

# --- подтверждение позиции / быстрый ретрай после входа ---
POSITION_CONFIRM_TIMEOUT   = 12.0   # раньше было 6
POSITION_POLL_INTERVAL     = 0.7
POST_ENTRY_TP_RETRY_WINDOW = 8.0    # секунд, пробуем дозавести TP/SL после входа
POST_ENTRY_TP_RETRY_SLEEP  = 0.8

# --- детекция TP (мягче по цене) ---
TP_DETECT_PRICE_STRICT = False      # False: сравниваем с допуском, а не "впритык"
TP_PRICE_TOL_PCT       = 0.0015     # 0.15% допуска по цене для распознавания TP

DO_NOT_CANCEL_EXISTING_TP = True    # не отменять чужие/ручные TP

# =========================
#      И Н И Ц И А Л И З А Ц И Я
# =========================
exchange = ccxt.bingx({
    'apiKey': BINGX_API_KEY,
    'secret': BINGX_SECRET_KEY,
    'options': {
        'defaultType': 'swap',
    },
    'enableRateLimit': True,
})

def load_markets_safe():
    for _ in range(2):
        try:
            return exchange.load_markets(reload=True)
        except Exception as e:
            print("Warning: load_markets:", e)
            time.sleep(1)
    return exchange.markets or {}

markets = load_markets_safe()

# Алиасы для ребрендов/нестандартных тикеров на BingX
SYMBOL_ALIASES = {
    'MATIC': 'POL',   # Polygon → POL
    'XBT':   'BTC',
}

def normalize_contract_symbol(sym: str) -> str:
    try:
        base = sym.split('/')[0].upper()
        if base in SYMBOL_ALIASES:
            new_base = SYMBOL_ALIASES[base]
            return sym.replace(base + '/USDT:USDT', new_base + '/USDT:USDT')
    except Exception:
        pass
    return sym

def _is_usdt_perp_market(m):
    try:
        return bool(m.get('contract')) and (m.get('quote') == 'USDT')
    except Exception:
        return False

def resolve_symbols(raw_list, markets_dict):
    resolved, skipped = [], []
    for s in raw_list:
        s_norm = normalize_contract_symbol(s)
        if s_norm in markets_dict:
            resolved.append(s_norm)
            continue
        alt = s_norm.replace(':USDT','')  # иногда без ':USDT'
        if alt in markets_dict:
            resolved.append(alt)
        else:
            skipped.append((s, s_norm))
    if skipped:
        print("[BOT] Пропущены неизвестные символы:", ", ".join([f"{a}->{b}" for a,b in skipped]))
    return resolved

def discover_symbols_top_n(n=30):
    """
    Собираем top-N USDT perp по объёму (quoteVolume) через fetch_tickers.
    Падать не будем: если что-то не получилось, вернём пустой список.
    """
    try:
        # кандидаты: все swap USDT из markets
        candidates = [m['symbol'] for m in markets.values() if _is_usdt_perp_market(m)]
        # уберём exotics/leveraged токены (часто имеют суффиксы)
        def _ok(sym):
            base = sym.split('/')[0].upper().replace(':USDT','')
            if base in UNIVERSE_EXCLUDE: return False
            # фильтр на очевидные 3L/3S/5L/5S/UP/DOWN и т.п.
            return not any(x in base for x in ('3L','3S','5L','5S','UP','DOWN'))
        candidates = [s for s in candidates if _ok(s)]
        if not candidates:
            return []

        tickers = exchange.fetch_tickers(candidates)
        scored = []
        for sym, t in tickers.items():
            try:
                qv = t.get('quoteVolume')
                if qv is None:
                    # fallback: baseVolume * last
                    bv = t.get('baseVolume')
                    last = t.get('last') or t.get('close')
                    if bv is not None and last is not None:
                        qv = float(bv) * float(last)
                if qv is None:
                    continue
                scored.append((sym, float(qv)))
            except Exception:
                continue

        scored.sort(key=lambda x: x[1], reverse=True)
        top = [s for s,_ in scored[:n]]
        return top
    except Exception as e:
        print("[BOT] discover_symbols_top_n warn:", e)
        return []

def build_universe():
    if UNIVERSE_MODE.upper() == "DYNAMIC":
        dyn = discover_symbols_top_n(UNIVERSE_TOP_N)
        if dyn:
            print(f"[BOT] DYNAMIC вселенная {len(dyn)} символов собрана.")
            return dyn
        print("[BOT] DYNAMIC не удался — fallback на STATIC.")
    # STATIC
    return resolve_symbols(SYMBOLS_TO_TRADE_STATIC, markets)

# финальный список торгуемых символов
SYMBOLS_TO_TRADE = build_universe()
print("[BOT] Торгуемые символы:", ", ".join(SYMBOLS_TO_TRADE))

# =========================
#      Х Е Л П Е Р Ы
# =========================
def is_contract(symbol):
    m = markets.get(symbol)
    return bool(m and m.get('contract'))

def get_market(symbol):
    m = markets.get(symbol)
    if m is None:
        load_markets_safe()
        m = markets.get(symbol)
    return m

def round_price(symbol, price):
    """Округление цены по правилам биржи (тик-сайз/precision) через ccxt."""
    try:
        return float(exchange.price_to_precision(symbol, price))
    except Exception:
        return float(price)

def floor_amount_by_precision(amount, precision):
    if precision is None:
        return float(amount)
    factor = 10 ** precision
    return math.floor(amount * factor) / factor

def round_amount(symbol, amount):
    """Округление количества по правилам биржи через ccxt."""
    try:
        return float(exchange.amount_to_precision(symbol, amount))
    except Exception:
        return float(amount)

def timeframe_seconds(tf):
    unit = tf[-1]; n = int(tf[:-1])
    if unit == 'm': return n * 60
    if unit == 'h': return n * 3600
    if unit == 'd': return n * 86400
    raise ValueError(f"Unsupported timeframe: {tf}")

def now_utc():
    return datetime.now(timezone.utc)

def utc_hour():
    return now_utc().hour

def ensure_json_log(line: dict):
    try:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
    except Exception as e:
        print("LOG write error:", e)

def wait_for_position_open(symbol, expect_long: bool, timeout_sec=POSITION_CONFIRM_TIMEOUT, poll_int=POSITION_POLL_INTERVAL):
    """
    Ждём, пока на бирже появится позиция по символу (contracts > 0).
    Возвращает (pos_dict | None).
    """
    t0 = time.time()
    while time.time() - t0 < timeout_sec:
        try:
            poss = fetch_positions_map()
            p = poss.get(symbol)
            if p:
                contracts = float(p.get('contracts') or 0.0)
                if contracts > 0 and ((contracts > 0) == expect_long):
                    return p
        except Exception:
            pass
        time.sleep(poll_int)
    return None

def get_symbol_limits(symbol):
    """Возвращает (min_amount, min_cost, amount_precision) из markets."""
    m = get_market(symbol)
    if not m:
        return None, None, None
    limits = m.get('limits', {}) or {}
    min_amount = (limits.get('amount') or {}).get('min')
    min_cost   = (limits.get('cost') or {}).get('min')
    amount_prec = m.get('precision', {}).get('amount')
    return min_amount, min_cost, amount_prec

def approx_equal(a, b, rel=1e-6, abs_=1e-8):
    return abs(a - b) <= max(rel * max(abs(a), abs(b)), abs_)

def fetch_open_orders_safe(symbol):
    try:
        return exchange.fetch_open_orders(symbol)
    except Exception as e:
        print(f"{symbol}: fetch_open_orders warn: {e}")
        return []

def is_tp_order(o, pos_side_long, tp_price=None):
    """TP: reduceOnly противоположной стороной; допускаем неточное совпадение цены."""
    try:
        if not o.get('reduceOnly'):
            return False
        side = (o.get('side') or '').lower()
        if pos_side_long and side != 'sell':
            return False
        if (not pos_side_long) and side != 'buy':
            return False
        typ = (o.get('type') or '').lower()
        # учтём типы TP некоторых бирж
        if typ not in ('limit', 'take_profit', 'take_profit_limit', 'takeProfit', 'takeProfitLimit'):
            # всё равно это может быть TP-лимит (биржи по-разному маркируют) — не отбрасываем по типу
            pass
        if tp_price is None:
            return True
        px = o.get('price')
        if px is None:
            return True  # Some CEX omit price in payload; не режем
        px = float(px); tp = float(tp_price)
        if TP_DETECT_PRICE_STRICT:
            return approx_equal(px, tp, rel=1e-5, abs_=1e-12)
        else:
            # допуск по относительному отклонению
            return abs(px - tp) / max(tp, 1e-12) <= TP_PRICE_TOL_PCT
    except Exception:
        return False

def ro_close_open_qty(symbol, pos_side_long):
    """
    Сумма reduceOnly на стороне закрытия; вернём (qty_sum, price_set, orders).
    """
    qty_sum = 0.0
    price_set = set()
    orders = []
    try:
        opens = fetch_open_orders_safe(symbol)
        for o in opens:
            if not o.get('reduceOnly'):
                continue
            side = (o.get('side') or '').lower()
            if pos_side_long and side != 'sell':
                continue
            if (not pos_side_long) and side != 'buy':
                continue
            rem = o.get('remaining'); amt = o.get('amount')
            q = float(rem if rem is not None else (amt or 0.0))
            qty_sum += max(0.0, q)
            px = o.get('price')
            if px is not None:
                try: price_set.add(float(px))
                except: pass
            orders.append(o)
    except Exception as e:
        print(f"{symbol}: ro_close_open_qty warn: {e}")
    return qty_sum, price_set, orders


def is_sl_order(o):
    """SL — стоп-ордер reduceOnly; ищем по type/params/info."""
    try:
        if not o.get('reduceOnly'): 
            return False
        typ = (o.get('type') or '').lower()
        if 'stop' in typ:
            return True
        p = o.get('params') or {}
        if any(k in p for k in ('stopPrice','triggerPrice','stopLossPrice')):
            return True
        inf = o.get('info') or {}
        if isinstance(inf, dict) and any(k in inf for k in ('stopPrice','triggerPrice','stopLossPrice')):
            return True
    except Exception:
        pass
    return False

def place_tp_order(symbol, pos_side_long, qty, tp_price):
    """Ставит reduceOnly лимит TP противоположной стороной."""
    try:
        side = 'sell' if pos_side_long else 'buy'
        price = round_price(symbol, tp_price)
        q = round_amount(symbol, qty)
        if q <= 0:
            return None
        return exchange.create_order(symbol, 'limit', side, q, price, {'reduceOnly': True})
    except Exception as e:
        print(f"{symbol}: place_tp_order warn: {e}")
        return None

def place_sl_order(symbol, pos_side_long, qty, sl_price):
    """Пытается поставить стоп-маркет reduceOnly разными параметрами."""
    last_err = None
    side = 'sell' if pos_side_long else 'buy'
    for tpl in SL_MARKET_PARAMS_LIST:
        try:
            params = dict(tpl)
            # подставим цену в нужный ключ
            if 'stopPrice' in params and params['stopPrice'] is None:
                params['stopPrice'] = round_price(symbol, sl_price)
            if 'triggerPrice' in params and params['triggerPrice'] is None:
                params['triggerPrice'] = round_price(symbol, sl_price)
            if 'stopLossPrice' in params and params['stopLossPrice'] is None:
                params['stopLossPrice'] = round_price(symbol, sl_price)
            q = round_amount(symbol, qty)
            if q <= 0:
                return None
            # тип 'market' у ccxt для стопов на некоторых биржах обязателен
            o = exchange.create_order(symbol, 'market', side, q, None, params)
            return o
        except Exception as e2:
            last_err = str(e2)
            continue
    print(f"{symbol}: place_sl_order fail, last err: {last_err}")
    return None

def replace_sl_order(symbol, pos_side_long, qty, new_sl_price):
    """Удаляет старые SL и ставит новый."""
    try:
        opens = fetch_open_orders_safe(symbol)
        # отменим старые SL
        for o in opens:
            if is_sl_order(o):
                try:
                    exchange.cancel_order(o['id'], symbol)
                except Exception:
                    pass
        return place_sl_order(symbol, pos_side_long, qty, new_sl_price)
    except Exception as e:
        print(f"{symbol}: replace_sl_order warn: {e}")
        return None

def order_book_imbalance(symbol, top_levels=PREPUMP_BOOK_TOP_LEVELS):
    """
    Возвращает (bid_notional, ask_notional, ratio) по топ-N уровням.
    """
    try:
        ob = exchange.fetch_order_book(symbol, limit=max(10, top_levels))
        bids = ob.get('bids') or []
        asks = ob.get('asks') or []
        b_sum = sum(float(p)*float(q) for p,q in bids[:top_levels])
        a_sum = sum(float(p)*float(q) for p,q in asks[:top_levels])
        ratio = (b_sum / a_sum) if a_sum > 0 else float('inf') if b_sum>0 else 1.0
        return b_sum, a_sum, ratio
    except Exception as e:
        print(f"{symbol}: order_book_imbalance warn: {e}")
        return None, None, 1.0

def open_interest_growth(symbol, df):
    """
    Опционально: прирост OI за последний бар (если биржа/ccxt поддерживают).
    Возвращает дробь роста (например, 0.07 = +7%) или None, если недоступно.
    """
    try:
        if not hasattr(exchange, "fetch_open_interest"):
            return None
        oi_now = exchange.fetch_open_interest(symbol)
        if not oi_now:
            return None
        # попробуем вытащить прошлое из df, если вы где-то храните; иначе — не сравниваем
        # тут заглушка: вернём None, чтобы не ломать логику (реализация историзации OI — отдельная задача)
        return None
    except Exception:
        return None

def is_volume_ramp_and_flat(df, lookback=PREPUMP_VOL_LOOKBACK, min_rising=PREPUMP_MIN_RISING_BARS, max_flat_pct=PREPUMP_MAX_FLAT_PCT):
    """
    Цена стоит во флэте, объёмы подрастают — «тихое накопление».
    """
    if len(df) < lookback + 1:
        return False, "Недостаточно баров"
    w = df.iloc[-lookback-1:-1]  # последние lookback закрытых баров
    vols = w['volume'].values
    closes = w['close'].values
    mx, mn = closes.max(), closes.min()
    flat = (mx - mn) / max(closes[-1], 1e-9) <= max_flat_pct
    rising = 0
    for i in range(1, len(vols)):
        if vols[i] > vols[i-1] and vols[i] > 0:
            rising += 1
    if flat and rising >= min_rising:
        return True, f"Флэт {(mx-mn)/closes[-1]:.4f} и растущий объём ({rising}/{len(vols)-1})"
    return False, "Нет накопления объёма во флэте"

def prepump_detector(symbol, df, regime):
    """
    Возвращает ('LONG', reason, context_dict) или (None, reason, context_dict).
    Критерии:
      - Накопление объёма при флэте
      - Перекос стакана в bid (bids >= ratio * asks)
      - (Опц.) Рост OI
      - Рын. режим не против LONG
      - Фандинг не убийственный (проверка будет в place_trade)
    """
    allow_long, _ = regime_filter(regime["btc_dominance"], regime["dxy_trend"], symbol)
    if not allow_long:
        return None, "Режим рынка против LONG (pre-pump)", {}

    ok_accum, why = is_volume_ramp_and_flat(df)
    if not ok_accum:
        return None, f"PrePump: {why}", {}

    b_sum, a_sum, ratio = order_book_imbalance(symbol)
    if ratio < PREPUMP_BID_ASK_RATIO_MIN:
        return None, f"PrePump: слабый дисбаланс стакана (ratio={ratio:.2f})", {"ob_ratio": ratio}

    oi_growth = open_interest_growth(symbol, df)
    # если есть данные OI — используем мягкий фильтр
    if oi_growth is not None and oi_growth < PREPUMP_OI_MIN_GROWTH_PCT:
        return None, f"PrePump: слабый рост OI ({oi_growth:.2%})", {"ob_ratio": ratio, "oi_growth": oi_growth}

    reason = f"PrePump: {why}; стакан bid/ask≈{ratio:.2f}"
    ctx = {"ob_ratio": ratio, "oi_growth": oi_growth}
    return "LONG", reason, ctx

def fetch_funding(symbol):
    """Вернёт dict {'rate': float, 'next_ts': ms} или None."""
    try:
        if exchange.has.get('fetchFundingRate'):
            fr = exchange.fetch_funding_rate(symbol)
        elif exchange.has.get('fetchFundingRates'):
            frs = exchange.fetch_funding_rates([symbol])
            # ccxt может вернуть dict или list
            if isinstance(frs, dict):
                fr = frs.get(symbol) or (list(frs.values())[0] if frs else None)
            elif isinstance(frs, list):
                fr = frs[0] if frs else None
            else:
                fr = None
        else:
            return None

        if not fr:
            return None

        rate = float(fr.get('fundingRate') or 0.0)
        nxt = fr.get('nextFundingTimestamp') or fr.get('nextFundingTime') \
              or (fr.get('info', {}).get('nextFundingTime') if isinstance(fr.get('info'), dict) else None)

        # приведи к int ms
        if isinstance(nxt, str):
            try:
                nxt = int(nxt)
            except Exception:
                nxt = None
        if isinstance(nxt, (int, float)):
            nxt = int(nxt)
        else:
            nxt = None

        return {"rate": rate, "next_ts": nxt}
    except Exception as e:
        print(f"{symbol}: fetch_funding warn: {e}")
        return None


def funding_risk_adjustment(symbol, side):
    """
    side: 'LONG' | 'SHORT'
    Возвращает (risk_multiplier, hold_reason, rate_float)
    - 0.0 => HOLD (запрет на вход)
    - 0.5 => снизить риск вдвое
    - 1.0 => без изменений
    """
    info = fetch_funding(symbol)
    if not info:
        return 1.0, None, None

    rate = info["rate"]
    next_ts = info["next_ts"]
    unfavorable = (side == 'LONG' and rate > 0) or (side == 'SHORT' and rate < 0)
    mult = 1.0
    reason = None

    # Блокировка за ~час до невыгодного funding
    if next_ts:
        secs = int(next_ts / 1000 - time.time())
        if secs < FUNDING_NO_OPEN_WINDOW_SEC and unfavorable and abs(rate) >= 0.0001:
            return 0.0, f"До funding ~{secs//60} мин, ставка {rate*100:.3f}% неблагоприятна — пропуск", rate

    # Жёсткий/мягкий пороги
    if unfavorable:
        if abs(rate) >= FUNDING_HARD_ABS:
            return 0.0, f"Funding {rate*100:.3f}% неблагоприятен (>= {FUNDING_HARD_ABS*100:.2f}%) — блокировка", rate
        if abs(rate) >= FUNDING_SOFT_ABS:
            mult = 0.5
            reason = f"Funding {rate*100:.3f}% неблагоприятен (>= {FUNDING_SOFT_ABS*100:.2f}%) — риск x0.5"

    return mult, reason, rate

def ro_close_qty_at_or_better(symbol, pos_side_long, threshold_price, tol_pct=TP_PRICE_TOL_PCT):
    """
    Считает суммарный reduceOnly объём на стороне закрытия,
    который стоит НЕ ХУЖЕ threshold_price (с допуском tol_pct).
    Для LONG: цена ордера >= threshold_price*(1 - tol_pct)
    Для SHORT: цена ордера <= threshold_price*(1 + tol_pct)
    Возвращает (qty_sum, best_price, orders_matched)
    """
    qty_sum = 0.0
    best_price = None
    matched = []
    try:
        opens = fetch_open_orders_safe(symbol)
        for o in opens:
            if not o.get('reduceOnly'):
                continue
            side = (o.get('side') or '').lower()
            if pos_side_long and side != 'sell':
                continue
            if (not pos_side_long) and side != 'buy':
                continue
            px = o.get('price')
            if px is None:
                # без цены считаем как подходящий (лучше не мешать)
                rem = o.get('remaining'); amt = o.get('amount')
                q = float(rem if rem is not None else (amt or 0.0))
                qty_sum += max(0.0, q)
                matched.append(o)
                continue
            try:
                pxf = float(px)
            except:
                continue

            if pos_side_long:
                # не хуже = НЕ НИЖЕ целевой (с небольшим допуском вниз)
                if pxf >= threshold_price * (1 - tol_pct):
                    rem = o.get('remaining'); amt = o.get('amount')
                    q = float(rem if rem is not None else (amt or 0.0))
                    qty_sum += max(0.0, q)
                    matched.append(o)
                    best_price = pxf if best_price is None else max(best_price, pxf)
            else:
                # для шорта не хуже = НЕ ВЫШЕ целевой
                if pxf <= threshold_price * (1 + tol_pct):
                    rem = o.get('remaining'); amt = o.get('amount')
                    q = float(rem if rem is not None else (amt or 0.0))
                    qty_sum += max(0.0, q)
                    matched.append(o)
                    best_price = pxf if best_price is None else min(best_price, pxf)
    except Exception as e:
        print(f"{symbol}: ro_close_qty_at_or_better warn: {e}")
    return qty_sum, best_price, matched


# =========================
#      Д А Н Н Ы Е
# =========================
def fetch_ohlcv(symbol, timeframe, limit=250):
    try:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not ohlcv:
            return pd.DataFrame()
        df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        return df
    except Exception as e:
        print(f"{symbol}: Ошибка fetch_ohlcv: {e}")
        return pd.DataFrame()

# =========================
#     И Н Д И К А Т О Р Ы
# =========================
def rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    sig = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - sig
    return macd_line, sig, hist

def atr(df, period=14):
    hl = df['high'] - df['low']
    hc = (df['high'] - df['close'].shift()).abs()
    lc = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def vwap(df, period=VWAP_PERIOD):
    tp = (df['high'] + df['low'] + df['close']) / 3.0
    pv = (tp * df['volume']).rolling(period, min_periods=period).sum()
    v  = df['volume'].rolling(period, min_periods=period).sum()
    vw = pv / v
    return vw

def keltner_channels(df, ema_period=EMA_FAST, atr_period=ATR_PERIOD, atr_mult=KELTNER_ATR_MULT):
    mid = df['close'].ewm(span=ema_period, adjust=False).mean()
    _atr = atr(df, period=atr_period)
    upper = mid + atr_mult * _atr
    lower = mid - atr_mult * _atr
    return upper, mid, lower

def volume_anomaly_flags(df, lookback=VOL_LOOKBACK, mult=VOL_ANOM_MULT):
    vol = df['volume']
    ma = vol.rolling(lookback, min_periods=lookback).mean()
    vol_std = vol.rolling(lookback, min_periods=lookback).std()
    flag = vol > (ma * mult)
    z = (vol - ma) / vol_std.replace(0, np.nan)
    return flag.fillna(False), z.fillna(0.0)

def add_indicators(df):
    if df.empty or len(df) < max(EMA_TREND_PERIOD, ATR_PERIOD, RSI_PERIOD, MACD_SLOW, VWAP_PERIOD, VOL_LOOKBACK) + 2:
        return pd.DataFrame()
    d = df.copy()
    d['RSI'] = rsi(d['close'], RSI_PERIOD)
    macd_line, sig, hist = macd(d['close'], MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    d['MACD'] = macd_line
    d['MACD_sig'] = sig
    d['MACD_hist'] = hist
    d['ATR'] = atr(d, ATR_PERIOD)
    d['EMA200'] = d['close'].ewm(span=EMA_TREND_PERIOD, adjust=False).mean()
    d['EMA20']  = d['close'].ewm(span=EMA_FAST, adjust=False).mean()
    d['EMA50']  = d['close'].ewm(span=EMA_SLOW, adjust=False).mean()
    # Новые индикаторы
    d['VWAP20'] = vwap(d, period=VWAP_PERIOD)
    kc_u, kc_m, kc_l = keltner_channels(d, ema_period=EMA_FAST, atr_period=ATR_PERIOD, atr_mult=KELTNER_ATR_MULT)
    d['KC_upper'], d['KC_mid'], d['KC_lower'] = kc_u, kc_m, kc_l
    vol_anom, vol_z = volume_anomaly_flags(d, lookback=VOL_LOOKBACK, mult=VOL_ANOM_MULT)
    d['VOL_ANOM'], d['VOL_Z'] = vol_anom, vol_z
    d.dropna(inplace=True)
    return d

# =========================
#   Р Ы Н О Ч Н Ы Й  Р Е Ж И М
# =========================
def fetch_market_regime():
    """
    Заглушка: подставь реальные данные (BTC dominance %, тренд DXY) при интеграции.
    Верни dict: {"btc_dominance": float, "dxy_trend": "up"|"down"|"flat"}
    """
    return {"btc_dominance": 52.0, "dxy_trend": "flat"}

def regime_filter(btc_dom, dxy_trend, symbol):
    base = symbol.split('/')[0].upper()
    is_major = base in ('BTC', 'ETH')
    favorable_alt_long = (btc_dom < 55.0) and (dxy_trend in ('flat', 'down'))
    unfavorable_alt_long = (btc_dom > 60.0) or (dxy_trend == 'up')

    allow_long = True
    allow_short = True

    if not is_major:
        if unfavorable_alt_long:
            allow_long = False
        # Шорты альтов разрешим, но можно ужесточить при явном risk-on
    return allow_long, allow_short

def regime_adjust_risk(base_risk, btc_dom, dxy_trend):
    if btc_dom > 60.0 or dxy_trend == "up":
        return max(base_risk * 0.5, 0.001)  # урезаем риск в risk-off
    if btc_dom < 50.0 and dxy_trend in ("flat", "down"):
        return min(base_risk * 1.2, 0.01)   # слегка повышаем в risk-on
    return base_risk

# =========================
#   Р А С Ч Е Т  П О З И Ц И И
# =========================
def compute_position_size(usdt_balance, last_price, sl_price, symbol, leverage=1, risk_per_trade=RISK_PER_TRADE):
    risk_usd = usdt_balance * risk_per_trade
    risk_usd *= (1 - 2 * FEE_PCT)  # вход+выход комиссии
    per_unit_risk = abs((last_price * (1 + SLIPPAGE_PCT)) - sl_price)
    if per_unit_risk <= 0:
        return 0.0, 0.0, 0.0

    raw_amount = risk_usd / per_unit_risk
    m = get_market(symbol)

    max_nominal = usdt_balance * MAX_NOTIONAL_PCT * leverage
    nominal = raw_amount * last_price
    if nominal > max_nominal:
        raw_amount = max_nominal / last_price
        nominal = raw_amount * last_price

    min_amount = m.get('limits', {}).get('amount', {}).get('min') if m else None
    min_cost   = m.get('limits', {}).get('cost', {}).get('min') if m else None
    if min_cost:
        raw_amount = max(raw_amount, min_cost / last_price)
        nominal = raw_amount * last_price
    if min_amount:
        raw_amount = max(raw_amount, min_amount)
        nominal = raw_amount * last_price

    amount_final = round_amount(symbol, raw_amount)
    if amount_final <= 0:
        return 0.0, 0.0, 0.0

    used_margin = nominal / max(leverage, 1)
    return amount_final, nominal, used_margin

# =========================
#    П О З И Ц И И / С О С Т О Я Н И Е
# =========================
def fetch_positions_map():
    out = {}
    try:
        if not exchange.has.get('fetchPositions'):
            return out
        poss = exchange.fetch_positions()
        for p in poss:
            sym = p.get('symbol')
            out[sym] = p
    except Exception:
        pass
    return out

def has_open_position(symbol, positions_map=None):
    try:
        if positions_map is None:
            positions_map = fetch_positions_map()
        p = positions_map.get(symbol)
        if not p:
            return False
        contracts = float(p.get('contracts') or 0)
        return abs(contracts) > 0
    except Exception:
        return False

def open_positions_count(positions_map=None):
    try:
        if positions_map is None:
            positions_map = fetch_positions_map()
        cnt = 0
        for p in positions_map.values():
            if abs(float(p.get('contracts') or 0)) > 0:
                cnt += 1
        return cnt
    except Exception:
        return 0

def set_leverage_safely(symbol):
    # Для BingX в one-way чаще требуется side="BOTH"
    try:
        exchange.set_leverage(LEVERAGE, symbol, {'side': 'BOTH'})
    except Exception as e:
        print(f"{symbol}: set_leverage warn: {e}")

# =========================
#   В С П О М .  П Р А В И Л А
# =========================
def late_breakout_guard(prev, atr_mult=LATE_BREAKOUT_ATR_MULT):
    kc_u, kc_l = float(prev['KC_upper']), float(prev['KC_lower'])
    close = float(prev['close'])
    atrv = float(prev['ATR'])
    vol_ok = bool(prev['VOL_ANOM']) or (float(prev['VOL_Z']) > 1.0)
    if (close > kc_u + atr_mult * atrv) or (close < kc_l - atr_mult * atrv):
        return vol_ok  # вход только если подтверждён объёмом
    return True

# =========================
#        Т О Р Г О В Л Я
# =========================
def place_trade(
    symbol,
    signal,
    df,
    usdt_balance_cached=None,
    positions_map=None,
    closed_bar=None,
    risk_mult_override=1.0,
    leverage_override=None,
    tag=None,
):
    """
    Market-вход + постановка TP/SL.
    - Ждём подтверждение позиции дольше (POSITION_CONFIRM_TIMEOUT) и делаем post-entry ретрай
    - TP докладываем только недостающий объём и только по не-худшей цене (at-or-better)
    - SL ставим reduceOnly, с несколькими попытками
    """
    m = get_market(symbol)
    is_swap = bool(m and m.get('contract'))

    # баланс
    if usdt_balance_cached is None:
        try:
            bal = exchange.fetch_balance()
            usdt_balance_cached = float(bal['total'].get('USDT', 0))
        except Exception as e:
            print(f"{symbol}: Не удалось получить баланс: {e}")
            return False

    last_price = float(df.iloc[-1]['close'])
    atr_val    = float(df.iloc[-1]['ATR'])

    # базовые уровни по модели
    if signal == 'LONG':
        sl = last_price - SL_ATR_MULT * atr_val
        tp = last_price + TP_ATR_MULT * atr_val
        side = 'buy'
    else:
        sl = last_price + SL_ATR_MULT * atr_val
        tp = last_price - TP_ATR_MULT * atr_val
        side = 'sell'

    # фильтр экстремальной волатильности
    if (atr_val / max(last_price, 1e-12)) > MAX_ATR_PCT:
        reason = f"ATR слишком велик ({atr_val/last_price:.3f}) — пропуск"
        print(f"{symbol}: {reason}")
        ensure_json_log({"ts": now_utc().isoformat(), "event": "HOLD", "symbol": symbol, "reason": reason})
        return False

    # funding-адаптация риска
    fund_mult, fund_reason, fund_rate = funding_risk_adjustment(symbol, 'LONG' if signal == 'LONG' else 'SHORT')
    if fund_mult == 0.0:
        print(f"{symbol}: {fund_reason}")
        ensure_json_log({"ts": now_utc().isoformat(), "event": "HOLD", "symbol": symbol,
                         "reason": f"FUNDING_BLOCK: {fund_reason}", "funding_rate": fund_rate})
        return False

    # расчёт риска (режим рынка + funding + пользовательский override)
    reg = fetch_market_regime()
    dyn_risk = regime_adjust_risk(RISK_PER_TRADE, reg["btc_dominance"], reg["dxy_trend"])
    dyn_risk *= fund_mult
    dyn_risk *= max(0.1, float(risk_mult_override))  # не даём опуститься ниже 0.1x

    leverage_for_calc = leverage_override if leverage_override is not None else (LEVERAGE if is_swap else 1)

    # размер позиции
    amount, nominal, used_margin = compute_position_size(
        usdt_balance_cached, last_price, sl, symbol,
        leverage=leverage_for_calc, risk_per_trade=dyn_risk
    )
    if amount <= 0:
        reason = "Невозможно вычислить размер позиции — пропуск"
        print(f"{symbol}: {reason}")
        ensure_json_log({"ts": now_utc().isoformat(), "event": "HOLD", "symbol": symbol, "reason": reason})
        return False

    sl_r = round_price(symbol, sl)
    tp_r = round_price(symbol, tp)

    print(f"{symbol}: Баланс={usdt_balance_cached:.2f} | price={last_price:.6f} | amount={amount} | "
          f"nominal≈{nominal:.2f} | margin≈{used_margin:.2f} | SL={sl_r} | TP={tp_r} | "
          f"swap={is_swap} | funding={('%.4f%%'%(fund_rate*100) if fund_rate is not None else 'N/A')}")

    # запрет второго входа в тот же контракт
    if is_swap and has_open_position(symbol, positions_map):
        reason = "Уже есть открытая позиция — пропуск входа"
        print(f"{symbol}: {reason}")
        ensure_json_log({"ts": now_utc().isoformat(), "event": "HOLD", "symbol": symbol, "reason": reason})
        return False

# режим/плечо для swap
    if is_swap:
        try:
            try:
                exchange.set_margin_mode('cross', symbol)
            except Exception as e:
                print(f"{symbol}: set_margin_mode warn: {e}")
            try:
                exchange.set_position_mode(False, symbol)  # one-way
            except Exception as e:
                print(f"{symbol}: set_position_mode warn: {e}")
            try:
                exchange.set_leverage(leverage_for_calc, symbol, {'side': 'BOTH'})
            except Exception as e:
                print(f"{symbol}: set_leverage warn: {e}")
        except Exception as e:
            print(f"{symbol}: init leverage/mode warn: {e}")

    # лог о подготовке
    entry_report = {
        "ts": now_utc().isoformat(),
        "event": "ENTRY_PREPARE",
        "mode": MODE,
        "symbol": symbol,
        "signal": signal,
        "amount": amount,
        "entry_price": last_price,
        "sl": sl_r,
        "tp": tp_r,
        "nominal": nominal,
        "used_margin": used_margin,
        "dry": DRY_RUN,
        "tag": tag,
        "funding_rate": fund_rate,
        "funding_risk_mult": fund_mult,
    }
    ensure_json_log(entry_report)

    # ——— ВХОД ———
    try:
        if DRY_RUN:
            print(f"{PRINT_PREFIX} [DRY_RUN] market {side} {amount} {symbol} (без SL/TP)")
            ensure_json_log({**entry_report, "event": "ENTRY_DRY"})
            print(f"{PRINT_PREFIX} [DRY_RUN] place reduce-only TP/SL after entry")
            return {"dry": True}

        order = exchange.create_order(symbol, 'market', side, amount)
        oid = order.get('id') if isinstance(order, dict) else None
        print(f"{symbol}: entry order id={oid or order}")
        ensure_json_log({**entry_report, "event": "ENTRY", "order_id": oid})

        # Ждём подтверждения позиции (дольше) + пост-ретрай
        pos_side_long = (signal == 'LONG')
        pos_info = wait_for_position_open(
            symbol, pos_side_long,
            timeout_sec=POSITION_CONFIRM_TIMEOUT,
            poll_int=POSITION_POLL_INTERVAL
        )

        if not pos_info:
            # короткий ретрай дозаведения в том же проходе
            t0 = time.time()
            while time.time() - t0 < POST_ENTRY_TP_RETRY_WINDOW:
                poss = fetch_positions_map()
                p = poss.get(symbol)
                if p:
                    contracts_now = abs(float(p.get('contracts') or 0.0))
                    if contracts_now > 0:
                        pos_info = p
                        break
                time.sleep(POST_ENTRY_TP_RETRY_SLEEP)

            if not pos_info:
                print(f"{symbol}: позиция не подтвердилась — TP/SL поставит manage_positions на следующем цикле")
                return True

        # ---------- ПОСТАНОВКА TP (top-up, at-or-better) ----------
        contracts_now = abs(float(pos_info.get('contracts') or 0.0))
        min_amt, min_cost, amt_prec = get_symbol_limits(symbol)

        # целевая цена TP по модели
        tp_price_model = tp_r

        # сколько TP уже стоит по НЕ ХУЖЕ цене
        qty_ok, best_px, _ = ro_close_qty_at_or_better(
            symbol, pos_side_long, tp_price_model, tol_pct=TP_PRICE_TOL_PCT
        )

        tp_qty_target = contracts_now * (TP_PART_FRACTION if TP_AS_PARTIAL else 1.0)
        need_tp_qty   = max(0.0, min(tp_qty_target - qty_ok, contracts_now))

        # не ухудшаем цену при дозаводе
        tp_price_eff = tp_price_model
        if best_px is not None:
            tp_price_eff = max(tp_price_model, best_px) if pos_side_long else min(tp_price_model, best_px)

        # учтём минималки
        if need_tp_qty > 0:
            if min_amt:
                need_tp_qty = max(need_tp_qty, float(min_amt))
            if min_cost:
                need_tp_qty = max(need_tp_qty, float(min_cost) / max(last_price, 1e-9))
        tp_qty = round_amount(symbol, need_tp_qty)

        # опциональный тик-паддинг
        mkt = get_market(symbol) or {}
        price_prec = (mkt.get('precision') or {}).get('price', 8)
        if TP_PRICE_PADDING_TICKS and price_prec is not None:
            tick = 10 ** (-price_prec)
            base = tp_price_eff
            tp_price_eff = round_price(
                symbol,
                base + (TP_PRICE_PADDING_TICKS * tick if pos_side_long else -TP_PRICE_PADDING_TICKS * tick)
            )

        if tp_qty > 0:
            _tp_ok = None
            for _ in range(3):  # несколько попыток на случай лагов API
                _tp_ok = place_tp_order(symbol, pos_side_long, tp_qty, tp_price_eff)
                if _tp_ok:
                    break
                time.sleep(0.6)
            if _tp_ok:
                print(f"{symbol}: TP set +{tp_qty} @ {tp_price_eff} reduceOnly (top-up, at-or-better; covered={qty_ok:.6f})")
                ensure_json_log({
                    "ts": now_utc().isoformat(), "event": "TP_ORDER", "symbol": symbol,
                    "amount": tp_qty, "price": tp_price_eff, "order_id": _tp_ok.get('id')
                })
            else:
                print(f"{symbol}: TP placement failed (после входа). Переставит менеджер.")
        else:
            print(f"{symbol}: TP already covered at-or-better (qty_ok={qty_ok:.6f}/{tp_qty_target:.6f})")
        # ---------- ПОСТАНОВКА SL (reduceOnly, с ретраями) ----------
        _sl_ok = None
        for _ in range(2):
            _sl_ok = place_sl_order(symbol, pos_side_long, contracts_now, sl_r)
            if _sl_ok:
                break
            time.sleep(0.6)
        if _sl_ok:
            print(f"{symbol}: SL set @ {sl_r} reduceOnly")
            ensure_json_log({
                "ts": now_utc().isoformat(), "event": "SL_ORDER", "symbol": symbol,
                "amount": contracts_now, "price": sl_r, "order_id": _sl_ok.get('id')
            })
        else:
            print(f"{symbol}: SL placement failed (после входа). Переставит менеджер.")

        return True

    except Exception as e:
        msg = str(e)
        print(f"{symbol}: ENTRY error: {msg}")
        ensure_json_log({**entry_report, "event": "ENTRY_ERROR", "error": msg})
        return False

#==========================
#      С И Г Н А Л Ы
# =========================
def impulse_signal(df, lookback=20, atr_k=0.2, body_frac_min=0.6, htf=None):
    """
    Импульсный вход с усилителями:
    - пробой экстремума на >= atr_k * ATR
    - тело >= body_frac_min от диапазона
    - Keltner подтверждает направление, VWAP не слишком далеко
    - желательно объёмная аномалия
    - тренд HTF в ту же сторону (если htf передан)
    """
    if len(df) < (lookback + 3):
        return "HOLD", "Недостаточно данных для импульса"

    prev = df.iloc[-2]
    atr  = float(prev['ATR'])
    rng  = float(prev['high'] - prev['low'])
    body = float(abs(prev['close'] - prev['open']))
    body_ok = (rng > 0) and (body / rng >= body_frac_min)

    recent_low  = df['low'].iloc[-lookback-2:-2].min()
    recent_high = df['high'].iloc[-lookback-2:-2].max()

    kc_u, kc_l = float(prev['KC_upper']), float(prev['KC_lower'])
    vwap20 = float(prev['VWAP20'])
    close = float(prev['close'])
    vol_anom = bool(prev['VOL_ANOM'])
    vol_z = float(prev['VOL_Z'])

    # HTF тренд
    htf_ok_long = htf_ok_short = True
    if htf is not None and len(htf) >= 3:
        h = htf.iloc[-2]
        htf_ok_long  = (h['EMA20'] > h['EMA50'] > h['EMA200']) and (h['MACD_hist'] > 0)
        htf_ok_short = (h['EMA20'] < h['EMA50'] < h['EMA200']) and (h['MACD_hist'] < 0)

    # LONG
    if (prev['close'] > recent_high + atr_k * atr) and body_ok and htf_ok_long:
        if (close >= kc_u) and ((close - vwap20) / vwap20 <= 0.02) and (vol_anom or vol_z > 1.0):
            if late_breakout_guard(prev):
                return "LONG", "Импульс: пробой + Keltner up + VWAP близко + объём"
            else:
                return "HOLD", "Поздний пробой без объёма"

    # SHORT
    if (prev['close'] < recent_low - atr_k * atr) and body_ok and htf_ok_short:
        if (close <= kc_l) and ((vwap20 - close) / vwap20 <= 0.02) and (vol_anom or vol_z > 1.0):
            if late_breakout_guard(prev):
                return "SHORT", "Импульс: пробой + Keltner low + VWAP близко + объём"
            else:
                return "HOLD", "Поздний пробой без объёма"

    return "HOLD", "Импульса нет"

def generate_signal(df, d_htf, symbol, regime):
    if len(df) < 3:
        return "HOLD", "Недостаточно данных"

    prev2 = df.iloc[-3]
    prev  = df.iloc[-2]

    # Pump alert: экстремальный выход за Keltner при сильной объёмной аномалии — пропускаем и лочим монету
    kc_span = float(prev['KC_upper'] - prev['KC_lower'])
    if kc_span > 0:
        dist_up = (float(prev['close']) - float(prev['KC_upper'])) / kc_span
        dist_dn = (float(prev['KC_lower']) - float(prev['close'])) / kc_span
        if (float(prev['VOL_Z']) > 2.5) and (dist_up > 0.8 or dist_dn > 0.8):
            return "HOLD", "Pump alert: экстремальный выход за Keltner + объём"

    # Базовая логика RSI-cross + MACD + расположение относительно VWAP/KC_mid
    def bullish_conf(p):
        above_vwap = float(p['close']) >= float(p['VWAP20'])
        above_kc_mid = float(p['close']) >= float(p['KC_mid'])
        macd_ok = p['MACD_hist'] > 0 and p['MACD'] > p['MACD_sig']
        return above_vwap and above_kc_mid and macd_ok

    def bearish_conf(p):
        below_vwap = float(p['close']) <= float(p['VWAP20'])
        below_kc_mid = float(p['close']) <= float(p['KC_mid'])
        macd_ok = p['MACD_hist'] < 0 and p['MACD'] < p['MACD_sig']
        return below_vwap and below_kc_mid and macd_ok

    if prev2['RSI'] < RSI_OVERSOLD and prev['RSI'] >= RSI_OVERSOLD and bullish_conf(prev):
        sig = "LONG"
        # Фильтр режима рынка
        allow_long, _ = regime_filter(regime["btc_dominance"], regime["dxy_trend"], symbol)
        if not allow_long:
            return "HOLD", "Фильтр режима: LONG запрещён для альтов в risk-off"
        if not late_breakout_guard(prev):
            return "HOLD", "Поздний пробой без объёма"
        return sig, "RSI выход + MACD + выше VWAP/KC_mid"

    if prev2['RSI'] > RSI_OVERBOUGHT and prev['RSI'] <= RSI_OVERBOUGHT and bearish_conf(prev):
        sig = "SHORT"
        _, allow_short = regime_filter(regime["btc_dominance"], regime["dxy_trend"], symbol)
        if not allow_short:
            return "HOLD", "Фильтр режима: SHORT запрещён (смягчи правило при необходимости)"
        if not late_breakout_guard(prev):
            return "HOLD", "Поздний пробой без объёма"
        return sig, "RSI выход + MACD + ниже VWAP/KC_mid"

    # LIVE: тренд-фильтр
    if MODE == "LIVE" and len(d_htf) >= 3:
        hprev = d_htf.iloc[-2]
        htf_ok_long  = hprev['close'] > hprev['EMA200'] and hprev['EMA20'] > hprev['EMA50'] > hprev['EMA200']
        htf_ok_short = hprev['close'] < hprev['EMA200'] and hprev['EMA20'] < hprev['EMA50'] < hprev['EMA200']
        # если нет подтверждения — шанс импульсу ниже

    # Импульсный вход (усиленный)
    if USE_IMPULSE:
        lb = LOOKBACK_BREAKOUT if MODE == "TEST" else max(LOOKBACK_BREAKOUT, 20)
        s2, r2 = impulse_signal(df, lookback=lb, atr_k=0.2, body_frac_min=0.6, htf=d_htf)
        if s2 != "HOLD":
            allow_long, allow_short = regime_filter(regime["btc_dominance"], regime["dxy_trend"], symbol)
            if s2 == "LONG" and not allow_long:
                return "HOLD", "Фильтр режима: LONG запрещён (импульс)"
            if s2 == "SHORT" and not allow_short:
                return "HOLD", "Фильтр режима: SHORT запрещён (импульс)"
            return s2, r2

    return "HOLD", "Нет конвергенции сигналов"

# =========================
#   У П Р А В Л Е Н И Е  П О З И Ц И Е Й
# =========================
def manage_positions(df_map):
    """
    - Контролируем открытые позиции.
    - Переносим SL в BE/Trail только в нашу сторону (никогда не назад).
    - TP выставляем/дополняем ТОЛЬКО по цене не хуже (at-or-better), не трогаем ручные TP.
    - SL ставим/обновляем: либо editOrder, либо через replace (отмена старых SL и новая заявка).
    """
    if DRY_RUN:
        return

    positions = fetch_positions_map()
    if not positions:
        return

    for symbol, p in positions.items():
        try:
            contracts = float(p.get('contracts') or 0.0)
            if abs(contracts) <= 0:
                continue

            m = df_map.get(symbol)
            if m is None or m.empty:
                continue

            side_long = (contracts > 0.0)
            entry = float(p.get('entryPrice') or 0.0) or float(p.get('markPrice') or 0.0)
            last  = float(m.iloc[-1]['close'])
            atr   = float(m.iloc[-1]['ATR'])

            # Базовые уровни по модели (от entry)
            base_tp = entry + TP_ATR_MULT * atr if side_long else entry - TP_ATR_MULT * atr
            base_sl = entry - SL_ATR_MULT * atr if side_long else entry + SL_ATR_MULT * atr
            tp_r    = round_price(symbol, base_tp)

            # Текущий SL, если биржа отдаёт
            sl_current = float(p.get('stopLossPrice') or 0.0)

            # Текущая доходность в R
            ref_sl_for_r = sl_current if sl_current != 0.0 else base_sl
            risk_per_unit = abs(entry - ref_sl_for_r)
            if risk_per_unit <= 0:
                risk_per_unit = max(abs(entry - base_sl), 1e-9)
            upnl_per_unit = (last - entry) if side_long else (entry - last)
            R_now = upnl_per_unit / risk_per_unit

            # Новый SL с учётом BE/Trail — только в сторону профита
            new_sl = base_sl
            if R_now >= BREAKEVEN_AT_R:
                new_sl = entry
            if R_now >= TRAIL_AT_R:
                trail_sl = (last - TRAIL_ATR_MULT * atr) if side_long else (last + TRAIL_ATR_MULT * atr)
                if side_long:
                    new_sl = max(new_sl, trail_sl)
                else:
                    new_sl = min(new_sl, trail_sl)
            new_sl_r = round_price(symbol, new_sl)

            # Открытые ордера для диагностики/проверок
            opens = fetch_open_orders_safe(symbol)

            # DEBUG-классификация (можно отключить позже)
            try:
                for o in opens:
                    side = (o.get('side') or '').lower()
                    typ  = (o.get('type') or '').lower()
                    px   = o.get('price')
                    ro   = bool(o.get('reduceOnly'))
                    is_tp = is_tp_order(o, side_long)
                    is_sl = is_sl_order(o)
                    if is_tp or is_sl:
                        print(f"{symbol}: open #{o.get('id')} side={side} type={typ} px={px} ro={ro} -> TP={is_tp} SL={is_sl}")
            except Exception:
                pass

            # ===== TP: дозаполнение ТОЛЬКО по цене "не хуже" =====
            # Сколько TP уже стоит "at-or-better":
            qty_ok, best_px, _ = ro_close_qty_at_or_better(
                symbol, side_long, tp_r, tol_pct=TP_PRICE_TOL_PCT
            )

            tp_qty_target = abs(contracts) * (TP_PART_FRACTION if TP_AS_PARTIAL else 1.0)
            need_qty = max(0.0, min(tp_qty_target - qty_ok, abs(contracts)))

            # Не ухудшаем цену при дозаводе
            tp_price_eff = tp_r
            if best_px is not None:
                tp_price_eff = max(tp_r, best_px) if side_long else min(tp_r, best_px)

            if ENFORCE_TP_SL and TP_REPLACE_IF_MISSED and need_qty > 0:
                # Биржевые минималки
                min_amt, min_cost, _ = get_symbol_limits(symbol)
                if min_amt:
                    need_qty = max(need_qty, float(min_amt))
                if min_cost:
                    need_qty = max(need_qty, float(min_cost) / max(last, 1e-9))
                need_qty = round_amount(symbol, need_qty)

                if need_qty > 0:
                    o = place_tp_order(symbol, side_long, need_qty, tp_price_eff)
                    if o:
                        print(f"{symbol}: TP_RENEW -> +{need_qty} @ {tp_price_eff} (at-or-better; covered={qty_ok:.6f}/{tp_qty_target:.6f})")
                        ensure_json_log({
                            "ts": now_utc().isoformat(),
                            "event": "TP_RENEW",
                            "symbol": symbol,
                            "amount": need_qty,
                            "price": tp_price_eff,
                            "order_id": o.get('id')
                        })
            else:
                # покрыто достаточно — ничего не делаем
                pass

            # ===== SL: если нет — ставим; если можно улучшить — двигаем только в сторону профита =====
            have_sl = any(is_sl_order(o) for o in opens)

            if ENFORCE_TP_SL and not have_sl and SL_REPLACE_IF_MISSED:
                o = place_sl_order(symbol, side_long, abs(contracts), new_sl_r)
                if o:
                    print(f"{symbol}: SL_RENEW -> @ {new_sl_r}")
                    ensure_json_log({
                        "ts": now_utc().isoformat(),
                        "event": "SL_RENEW",
                        "symbol": symbol,
                        "price": new_sl_r,
                        "order_id": o.get('id')
                    })

            # Улучшение SL (только в "плюс")
            need_move = (
                sl_current == 0.0 or
                (side_long and new_sl_r > sl_current + 1e-12) or
                ((not side_long) and new_sl_r < sl_current - 1e-12)
            )
            if need_move:
                try:
                    if SL_EDIT_VIA_EDITORDER and exchange.has.get('editOrder'):
                        patched = exchange.edit_order(None, symbol, None, None, None, {'stopLossPrice': new_sl_r})
                        print(f"{symbol}: MOVE_SL via editOrder -> {new_sl_r} | resp: {patched}")
                    else:
                        replace_sl_order(symbol, side_long, abs(contracts), new_sl_r)
                        print(f"{symbol}: MOVE_SL via replace -> {new_sl_r}")

                    ensure_json_log({
                        "ts": now_utc().isoformat(),
                        "event": "MOVE_SL",
                        "symbol": symbol,
                        "entry": entry,
                        "last": last,
                        "old_sl": sl_current,
                        "new_sl": new_sl_r,
                        "R_now": R_now
                    })
                except Exception as e:
                    print(f"{symbol}: manage SL warn: {e}")

        except Exception as e:
            print(f"{symbol}: manage_positions item warn: {e}")

# =========================
#         М А И Н
# =========================
def align_to_next_candle(tf_seconds):
    now = int(time.time())
    wait = tf_seconds - (now % tf_seconds) + 2
    print(f"\nЖдём до закрытия следующей свечи: ~{wait}s")
    time.sleep(wait)

state = {
    "day": None,
    "eq_start": None,
    "last_trade_bar_ts": {},
    "pump_lock": {},           # symbol -> bars_left
    "sl_orders": {},           # symbol -> last SL order id
    "tp_orders": {},           # symbol -> last TP order id
}

def update_pump_lock(symbol, pumped):
    if pumped:
        state["pump_lock"][symbol] = PUMP_LOCK_BARS
    else:
        left = state["pump_lock"].get(symbol, 0)
        if left > 0:
            state["pump_lock"][symbol] = left - 1

def is_locked(symbol):
    return state.get("pump_lock", {}).get(symbol, 0) > 0

def can_trade_today(usdt_eq):
    today = date.fromtimestamp(time.time())
    if state['day'] != today:
        state['day'] = today
        state['eq_start'] = usdt_eq
        print(f"{PRINT_PREFIX} Новый торговый день: equity start = {usdt_eq:.2f} USDT | MODE={MODE}")
        return True
    dd = (state['eq_start'] - usdt_eq) / max(state['eq_start'], 1)
    if dd >= MAX_DAILY_DD:
        print(f"{PRINT_PREFIX} Достигнут дневной стоп {dd*100:.2f}% >= {MAX_DAILY_DD*100:.2f}% — торговля пауза")
        return False
    return True

def cooldown_ok(symbol, df):
    last_close_ts = df.iloc[-2]['timestamp']
    lt = state['last_trade_bar_ts'].get(symbol)
    if lt is None:
        return True
    closed = df[df['timestamp'] > lt]
    if len(closed) >= TRADE_COOLDOWN_BARS:
        return True
    print(f"{symbol}: кулдаун {len(closed)}/{TRADE_COOLDOWN_BARS} — пропуск")
    return False

def main():
    print("Бот запущен (swap):", ", ".join(SYMBOLS_TO_TRADE), "| MODE:", MODE)
    tf_sec = timeframe_seconds(TIMEFRAME_ENTRY)
    last_closed_ts = {s: None for s in SYMBOLS_TO_TRADE}

    for s in SYMBOLS_TO_TRADE:
        if is_contract(s):
            try:
                exchange.set_margin_mode('cross', s)
                exchange.set_position_mode(False, s)  # one-way
                set_leverage_safely(s)
            except Exception as e:
                print(f"{s}: init leverage/mode warn: {e}")

    while True:
        try:
            try:
                bal = exchange.fetch_balance()
                usdt_balance = float(bal['total'].get('USDT', 0))
            except Exception as e:
                print("Не удалось получить баланс:", e)
                usdt_balance = None

            pos_map = fetch_positions_map()

            if usdt_balance is not None and not can_trade_today(usdt_balance):
                align_to_next_candle(tf_sec)
                continue

            if AVOID_HOURS_UTC and utc_hour() in AVOID_HOURS_UTC:
                reason = f"{PRINT_PREFIX} Вне торгового окна (UTC час={utc_hour()}) — пауза до следующей свечи"
                print(reason)
                ensure_json_log({"ts": now_utc().isoformat(), "event": "HOLD", "reason": reason})
                align_to_next_candle(tf_sec)
                continue

            # Фон рынка (один раз на цикл)
            regime = fetch_market_regime()

            df_for_manage = {}

            for symbol in SYMBOLS_TO_TRADE:
                print("\n" + "="*72)
                print(f"[{now_utc().strftime('%Y-%m-%d %H:%M:%S')} UTC] Анализ {symbol}")

                if is_locked(symbol):
                    left = state["pump_lock"].get(symbol, 0)
                    reason = f"{symbol}: pump lock активен ещё {left} бар(ов) — пропуск"
                    print(reason)
                    ensure_json_log({"ts": now_utc().isoformat(), "event": "HOLD", "symbol": symbol, "reason": reason})
                    update_pump_lock(symbol, pumped=False)  # тик вниз
                    continue

                df_raw = fetch_ohlcv(symbol, TIMEFRAME_ENTRY, limit=400)
                d_htf_raw = fetch_ohlcv(symbol, TIMEFRAME_HTF, limit=400)
                if df_raw.empty or d_htf_raw.empty:
                    reason = f"{symbol}: нет данных"
                    print(reason)
                    ensure_json_log({"ts": now_utc().isoformat(), "event": "HOLD", "symbol": symbol, "reason": reason})
                    continue

                df = add_indicators(df_raw)
                d_htf = add_indicators(d_htf_raw)
                if df.empty or d_htf.empty:
                    reason = f"{symbol}: недостаточно данных для индикаторов"
                    print(reason)
                    ensure_json_log({"ts": now_utc().isoformat(), "event": "HOLD", "symbol": symbol, "reason": reason})
                    continue

                closed_bar = df.iloc[-2]
                if last_closed_ts.get(symbol) == closed_bar['timestamp']:
                    reason = f"{symbol}: свеча не обновилась — пропуск"
                    print(reason)
                    ensure_json_log({"ts": now_utc().isoformat(), "event": "HOLD", "symbol": symbol, "reason": reason})
                    continue
                last_closed_ts[symbol] = closed_bar['timestamp']

                df_for_manage[symbol] = df

                if open_positions_count(pos_map) >= MAX_OPEN_POS:
                    reason = f"{PRINT_PREFIX} Достигнут лимит открытых позиций ({MAX_OPEN_POS}) — пропуск входов"
                    print(reason)
                    ensure_json_log({"ts": now_utc().isoformat(), "event": "HOLD", "symbol": symbol, "reason": reason})
                    continue

                if not cooldown_ok(symbol, df):
                    ensure_json_log({"ts": now_utc().isoformat(), "event": "HOLD", "symbol": symbol, "reason": "cooldown"})
                    continue

                signal, reason = generate_signal(df, d_htf, symbol, regime)
                print(f">>> {symbol} | Сигнал: {signal} | Причина: {reason}")

                pumped = ("Pump alert" in reason)
                update_pump_lock(symbol, pumped=pumped)

                if signal != "HOLD":
                    ok = place_trade(symbol, signal, df, usdt_balance_cached=usdt_balance, positions_map=pos_map, closed_bar=closed_bar)
                    print(f"{symbol}: place_trade -> {ok}")
                    if ok:
                        state['last_trade_bar_ts'][symbol] = closed_bar['timestamp']
                        ensure_json_log({
                            "ts": now_utc().isoformat(),
                            "event": "ENTRY_SIGNAL",
                            "mode": MODE,
                            "symbol": symbol,
                            "signal": signal,
                            "reason": reason
                        })
                else:
                    # --- PrePump Detector ---
                    if PREPUMP_ENABLE:
                        pp_sig, pp_reason, pp_ctx = prepump_detector(symbol, df, regime)
                        if pp_sig == "LONG":
                            print(f">>> {symbol} | PrePump ALERT | {pp_reason} | ctx={pp_ctx}")
                            ensure_json_log({
                                "ts": now_utc().isoformat(),
                                "event": "PREPUMP_ALERT",
                                "symbol": symbol,
                                "reason": pp_reason,
                                "ctx": pp_ctx
                            })
                            if PREPUMP_ENTER:
                                # Рассчитываем риск override: хотим PREPUMP_MAX_RISK вместо базового RISK_PER_TRADE
                                risk_mult = min(PREPUMP_MAX_RISK / max(RISK_PER_TRADE, 1e-9), 1.0)
                                ok = place_trade(
                                    symbol, "LONG", df,
                                    usdt_balance_cached=usdt_balance,
                                    positions_map=pos_map,
                                    closed_bar=closed_bar,
                                    risk_mult_override=risk_mult,
                                    leverage_override=PREPUMP_LEVERAGE,
                                    tag="PREPUMP"
                                )
                                print(f"{symbol}: prepump place_trade -> {ok}")
                                if ok:
                                    state['last_trade_bar_ts'][symbol] = closed_bar['timestamp']
                                    ensure_json_log({
                                        "ts": now_utc().isoformat(),
                                        "event": "PREPUMP_ENTRY",
                                        "mode": MODE,
                                        "symbol": symbol,
                                        "signal": "LONG",
                                        "reason": pp_reason,
                                        "ctx": pp_ctx
                                    })
                          # если вход не удался — просто продолжаем
                    # если pre-pump не сработал — пойдём на обычный HOLD
                    ensure_json_log({
                        "ts": now_utc().isoformat(),
                        "event": "HOLD",
                        "mode": MODE,
                        "symbol": symbol,
                        "reason": reason
                    })

            manage_positions(df_for_manage)
            align_to_next_candle(tf_sec)

        except KeyboardInterrupt:
            print("Остановлен пользователем.")
            break
        except Exception as e:
            print("Loop error:", e)
            time.sleep(15)

if __name__ == "__main__":
    main()
