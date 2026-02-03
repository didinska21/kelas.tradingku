import os
import time
import ccxt
import numpy as np
import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# ========== KONFIGURASI DARI .ENV ==========
TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
GROQ_API_KEY = os.getenv('GROQ_API_KEY')
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY')
CRYPTOPANIC_API_KEY = os.getenv('CRYPTOPANIC_API_KEY')

# Inisialisasi Groq Client
groq_client = Groq(api_key=GROQ_API_KEY)

# Inisialisasi Exchange Binance
exchange_spot = ccxt.binance({
    'apiKey': BINANCE_API_KEY,
    'secret': BINANCE_SECRET_KEY,
    'enableRateLimit': True,
    'options': {'defaultType': 'spot'}
})

exchange_futures = ccxt.binance({
    'apiKey': BINANCE_API_KEY,
    'secret': BINANCE_SECRET_KEY,
    'enableRateLimit': True,
    'options': {'defaultType': 'future'}
})

# ========== CACHE & COOLDOWN ==========
PAIR_CACHE = {'spot': [], 'futures': [], 'last_update': None}
USER_COOLDOWN = {}
COOLDOWN_SECONDS = 12

# ========== FUNGSI AMBIL SEMUA PAIR (DENGAN CACHE) ==========
def get_all_pairs(market_type='spot'):
    """Ambil semua pair USDT dari Binance dengan caching 10 menit"""
    now = datetime.now()
    # Kalau cache masih fresh (< 10 menit), pakai cache
    if (PAIR_CACHE[market_type] and PAIR_CACHE['last_update']
            and (now - PAIR_CACHE['last_update']) < timedelta(minutes=10)):
        return PAIR_CACHE[market_type]

    try:
        exchange = exchange_spot if market_type == 'spot' else exchange_futures
        markets = exchange.load_markets()

        usdt_pairs = []
        for symbol, market in markets.items():
            if market['quote'] == 'USDT' and market['active']:
                if market_type == 'spot' and market.get('spot', False):
                    usdt_pairs.append(symbol)
                elif market_type == 'futures' and (market.get('swap', False) or market.get('future', False)):
                    usdt_pairs.append(symbol)

        usdt_pairs.sort()

        # Update cache
        PAIR_CACHE[market_type] = usdt_pairs
        PAIR_CACHE['last_update'] = now

        print(f"✅ Loaded {len(usdt_pairs)} {market_type.upper()} pairs")
        return usdt_pairs

    except Exception as e:
        print(f"Error loading pairs: {e}")
        return [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT',
            'SOL/USDT', 'ADA/USDT', 'DOGE/USDT', 'AVAX/USDT'
        ]

# ========== FUNGSI AMBIL DATA OHLCV ==========
def get_ohlcv_data(symbol, market_type='spot', timeframe='5m', limit=100):
    """Ambil data OHLCV dari Binance"""
    try:
        exchange = exchange_spot if market_type == 'spot' else exchange_futures
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        return ohlcv
    except Exception as e:
        print(f"Error fetching OHLCV [{timeframe}]: {e}")
        return None

# ========== FUNGSI AMBIL NEWS DARI CRYPTOPANIC ==========
def get_crypto_news(symbol):
    """Ambil news dari CryptoPanic berdasarkan symbol"""
    try:
        # Ubah format symbol: BTC/USDT -> btc
        coin = symbol.split('/')[0].lower()

        url = "https://api.cryptopanic.com/v1/posts/"
        params = {
            'auth_token': CRYPTOPANIC_API_KEY,
            'currencies': coin.upper(),
            'kind': 'news',
            'limit': 5
        }
        response = requests.get(url, params=params, timeout=10)
        data = response.json()

        news_list = []
        if 'results' in data:
            for item in data['results'][:5]:
                news_list.append({
                    'title': item.get('title', 'N/A'),
                    'url': item.get('url', ''),
                    'published_at': item.get('published_at', ''),
                    'votes': item.get('votes', {})
                })
        return news_list

    except Exception as e:
        print(f"Error fetching news: {e}")
        return []

def format_news_for_prompt(news_list):
    """Format news untuk dikirim ke AI prompt"""
    if not news_list:
        return "Tidak ada news terbaru yang ditemukan."

    formatted = ""
    for i, news in enumerate(news_list, 1):
        # Hitung sentiment dari votes
        votes = news.get('votes', {})
        positive = votes.get('positive', 0)
        negative = votes.get('negative', 0)
        if positive > negative:
            sentiment = "BULLISH ✅"
        elif negative > positive:
            sentiment = "BEARISH ⚠️"
        else:
            sentiment = "NETRAL ➡️"

        formatted += f"{i}. [{sentiment}] {news['title']}\n"

    return formatted

def format_news_for_telegram(news_list):
    """Format news untuk ditampilkan di Telegram"""
    if not news_list:
        return "📰 Tidak ada news terbaru untuk coin ini."

    msg = "📰 *NEWS TERBARU:*\n"
    for i, news in enumerate(news_list, 1):
        votes = news.get('votes', {})
        positive = votes.get('positive', 0)
        negative = votes.get('negative', 0)
        if positive > negative:
            sentiment = "🟢 Bullish"
        elif negative > positive:
            sentiment = "🔴 Bearish"
        else:
            sentiment = "🟡 Netral"

        msg += f"\n{i}. {sentiment}\n"
        msg += f"   {news['title']}\n"

    return msg

# ========== FUNGSI HITUNG INDIKATOR TEKNIKAL ==========
def calculate_indicators(ohlcv_data):
    """Hitung semua indikator teknikal untuk scalping"""
    closes = np.array([x[4] for x in ohlcv_data], dtype=float)
    highs = np.array([x[2] for x in ohlcv_data], dtype=float)
    lows  = np.array([x[3] for x in ohlcv_data], dtype=float)
    volumes = np.array([x[5] for x in ohlcv_data], dtype=float)

    # ----- Moving Averages (rolling, ambil nilai terakhir) -----
    ma20 = float(np.convolve(closes, np.ones(20)/20, mode='valid')[-1]) if len(closes) >= 20 else float(closes[-1])
    ma50 = float(np.convolve(closes, np.ones(50)/50, mode='valid')[-1]) if len(closes) >= 50 else float(closes[-1])

    # ----- Support & Resistance (dari 20 candle terakhir) -----
    support   = float(np.min(lows[-20:]))
    resistance = float(np.max(highs[-20:]))

    # ----- Trend -----
    if closes[-1] > ma20 > ma50:
        trend = "Uptrend (Naik) 📈"
    elif closes[-1] < ma20 < ma50:
        trend = "Downtrend (Turun) 📉"
    else:
        trend = "Sideways (Datar) ↔️"

    # ----- RSI 14 (PERBAIKAN: pakai data terakhir, bukan pertama) -----
    deltas = np.diff(closes)
    seed = deltas[-14:]  # 14 candle TERAKHIR
    up   = seed[seed >= 0].sum() / 14
    down = -seed[seed < 0].sum() / 14
    rs   = up / down if down != 0 else 0
    rsi  = 100 - (100 / (1 + rs))

    # ----- MACD (12, 26, 9) -----
    if len(closes) >= 26:
        ema12 = float(closes[-1])  # seed
        ema26 = float(closes[-1])
        alpha12 = 2.0 / (12 + 1)
        alpha26 = 2.0 / (26 + 1)
        for c in closes[-26:]:
            ema12 = alpha12 * c + (1 - alpha12) * ema12
            ema26 = alpha26 * c + (1 - alpha26) * ema26
        macd_line   = ema12 - ema26
        # Signal line (EMA 9 dari MACD) — simplified: pakai rata-rata 9 nilai terakhir
        # Untuk akurasi lebih, buat array MACD terakhir
        macd_values = []
        ema12_t = float(closes[0])
        ema26_t = float(closes[0])
        for c in closes:
            ema12_t = alpha12 * c + (1 - alpha12) * ema12_t
            ema26_t = alpha26 * c + (1 - alpha26) * ema26_t
            macd_values.append(ema12_t - ema26_t)
        macd_values = np.array(macd_values)
        # Signal = EMA 9 dari macd_values
        alpha9 = 2.0 / (9 + 1)
        signal = macd_values[0]
        for m in macd_values:
            signal = alpha9 * m + (1 - alpha9) * signal
        macd_signal = float(signal)
        macd_hist   = float(macd_line - macd_signal)
    else:
        macd_line   = 0.0
        macd_signal = 0.0
        macd_hist   = 0.0

    # ----- Bollinger Bands (20 period, 2 std) -----
    if len(closes) >= 20:
        bb_mid   = float(np.mean(closes[-20:]))
        bb_std   = float(np.std(closes[-20:], ddof=0))
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
    else:
        bb_mid   = float(closes[-1])
        bb_upper = bb_mid
        bb_lower = bb_mid

    # ----- Fibonacci Retracement & Extension -----
    # Swing High & Swing Low dari 20 candle terakhir
    swing_high = float(np.max(highs[-20:]))
    swing_low  = float(np.min(lows[-20:]))
    fib_range  = swing_high - swing_low

    fib_levels = {
        '0.000': swing_low,
        '0.236': swing_low + fib_range * 0.236,
        '0.382': swing_low + fib_range * 0.382,
        '0.500': swing_low + fib_range * 0.500,
        '0.618': swing_low + fib_range * 0.618,
        '0.786': swing_low + fib_range * 0.786,
        '1.000': swing_high,
        # Extension levels
        '1.272': swing_low + fib_range * 1.272,
        '1.618': swing_low + fib_range * 1.618,
    }

    # Cari level Fib terdekat dari harga sekarang
    current_price = float(closes[-1])
    nearest_fib = min(fib_levels.items(), key=lambda x: abs(x[1] - current_price))

    # Price change
    price_change = ((closes[-1] - closes[-24]) / closes[-24] * 100) if len(closes) >= 24 else 0.0

    return {
        'current_price': current_price,
        'ma20': ma20,
        'ma50': ma50,
        'support': support,
        'resistance': resistance,
        'trend': trend,
        'rsi': float(rsi),
        'macd_line': macd_line,
        'macd_signal': macd_signal,
        'macd_hist': macd_hist,
        'bb_upper': bb_upper,
        'bb_mid': bb_mid,
        'bb_lower': bb_lower,
        'fib_levels': fib_levels,
        'nearest_fib': nearest_fib,
        'swing_high': swing_high,
        'swing_low': swing_low,
        'price_change': float(price_change),
        'closes': closes,
        'highs': highs,
        'lows': lows,
        'volume': float(np.mean(volumes[-20:]))
    }

# ========== FUNGSI BUAT DUAL TIMEFRAME CHART ==========
def create_dual_chart(ohlcv_5m, ohlcv_15m, ind_5m, ind_15m, symbol, market_type):
    """Buat chart dual timeframe: 15m (atas) + 5m (bawah) dengan Fibonacci & BB"""
    plt.style.use('dark_background')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), gridspec_kw={'height_ratios': [1, 1]})
    fig.patch.set_facecolor('#0e1117')

    market_label = "SPOT" if market_type == 'spot' else "FUTURES"

    # ---------- Helper: plot satu timeframe ----------
    def plot_tf(ax, ohlcv, ind, tf_label):
        timestamps = [x[0] for x in ohlcv]
        closes = ind['closes']
        dates  = [datetime.fromtimestamp(ts / 1000) for ts in timestamps]

        # Price line
        ax.plot(dates, closes, color='#00ff00', linewidth=2, zorder=3, label='Harga')

        # MA20 rolling
        if len(closes) >= 20:
            ma20_arr = np.convolve(closes, np.ones(20)/20, mode='valid')
            ma20_dates = dates[19:]
            ax.plot(ma20_dates, ma20_arr, color='#FFD700', linestyle='--', linewidth=1.5, label='MA20', zorder=2)

        # MA50 rolling
        if len(closes) >= 50:
            ma50_arr = np.convolve(closes, np.ones(50)/50, mode='valid')
            ma50_dates = dates[49:]
            ax.plot(ma50_dates, ma50_arr, color='#FF6347', linestyle='--', linewidth=1.5, label='MA50', zorder=2)

        # Bollinger Bands
        if len(closes) >= 20:
            bb_mid_arr   = np.convolve(closes, np.ones(20)/20, mode='valid')
            bb_std_arr   = np.array([np.std(closes[i:i+20], ddof=0) for i in range(len(closes)-19)])
            bb_upper_arr = bb_mid_arr + 2 * bb_std_arr
            bb_lower_arr = bb_mid_arr - 2 * bb_std_arr
            bb_dates     = dates[19:]
            ax.plot(bb_dates, bb_upper_arr, color='#8888ff', linestyle=':', linewidth=1, label='BB Upper')
            ax.plot(bb_dates, bb_lower_arr, color='#8888ff', linestyle=':', linewidth=1, label='BB Lower')
            ax.fill_between(bb_dates, bb_lower_arr, bb_upper_arr, alpha=0.08, color='#8888ff')

        # Fibonacci levels (horizontal dashed)
        fib_colors = {
            '0.236': '#aaaaaa', '0.382': '#ffaa00', '0.500': '#ffffff',
            '0.618': '#00ccff', '0.786': '#ff44aa',
            '1.272': '#44ff44', '1.618': '#44ff44'
        }
        for level_name, level_val in ind['fib_levels'].items():
            if level_name in fib_colors:
                ax.axhline(y=level_val, color=fib_colors[level_name], linestyle='--', linewidth=0.8, alpha=0.6)
                ax.text(dates[2], level_val, f'  Fib {level_name}', color=fib_colors[level_name],
                        fontsize=7, va='center', alpha=0.9)

        # Support & Resistance
        ax.axhline(y=ind['support'],    color='#00BFFF', linestyle='-', linewidth=2, alpha=0.8, label=f"Support ${ind['support']:.2f}")
        ax.axhline(y=ind['resistance'], color='#FF1493', linestyle='-', linewidth=2, alpha=0.8, label=f"Resist  ${ind['resistance']:.2f}")

        # Current price label
        ax.text(dates[-1], ind['current_price'], f"  ${ind['current_price']:.2f}",
                color='white', fontsize=11, va='center', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#00aa44', alpha=0.85))

        # Styling
        ax.set_title(f'{symbol} — {tf_label} | {market_label}', fontsize=14, fontweight='bold', color='white', pad=10)
        ax.set_ylabel('Harga (USDT)', fontsize=10, color='white')
        ax.legend(loc='upper left', fontsize=7, framealpha=0.7, ncol=3)
        ax.grid(True, alpha=0.2, linestyle=':')
        ax.set_facecolor('#0e1117')
        ax.tick_params(colors='white')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M'))
        fig.autofmt_xdate()

    # Plot kedua timeframe
    plot_tf(ax1, ohlcv_15m, ind_15m, '15 Menit (Trend)')
    plot_tf(ax2, ohlcv_5m,  ind_5m,  '5 Menit (Entry)')

    plt.tight_layout(pad=2.0)

    chart_path = f'chart_{symbol.replace("/", "_")}_{market_type}_scalp.png'
    plt.savefig(chart_path, dpi=130, bbox_inches='tight', facecolor='#0e1117')
    plt.close()
    return chart_path

# ========== FUNGSI ANALISA AI GROQ KHUSUS SCALPING ==========
def analyze_with_groq_scalping(symbol, ind_5m, ind_15m, news_text, market_type):
    """AI Groq analisis lengkap untuk scalping: teknikal + Fibonacci + News"""

    market_label = "SPOT" if market_type == 'spot' else "FUTURES"

    prompt = f"""
Kamu adalah trader profesional yang ahli dalam SCALPING crypto.
Analisa secara LENGKAP dan DETAIL pair {symbol} ({market_label}) untuk strategi SCALPING.

===== DATA TIMEFRAME 15 MENIT (TREND) =====
- Harga: ${ind_15m['current_price']:.4f}
- Trend: {ind_15m['trend']}
- RSI: {ind_15m['rsi']:.1f}
- MA20: ${ind_15m['ma20']:.4f}
- MA50: ${ind_15m['ma50']:.4f}
- MACD Line: {ind_15m['macd_line']:.6f} | Signal: {ind_15m['macd_signal']:.6f} | Histogram: {ind_15m['macd_hist']:.6f}
- Bollinger Bands: Upper ${ind_15m['bb_upper']:.4f} | Mid ${ind_15m['bb_mid']:.4f} | Lower ${ind_15m['bb_lower']:.4f}
- Support: ${ind_15m['support']:.4f} | Resistance: ${ind_15m['resistance']:.4f}
- Fibonacci Level Terdekat: {ind_15m['nearest_fib'][0]} di ${ind_15m['nearest_fib'][1]:.4f}
- Swing High: ${ind_15m['swing_high']:.4f} | Swing Low: ${ind_15m['swing_low']:.4f}
- Fib Levels: 0.382=${ind_15m['fib_levels']['0.382']:.4f} | 0.500=${ind_15m['fib_levels']['0.500']:.4f} | 0.618=${ind_15m['fib_levels']['0.618']:.4f} | 0.786=${ind_15m['fib_levels']['0.786']:.4f}
- Fib Extension: 1.272=${ind_15m['fib_levels']['1.272']:.4f} | 1.618=${ind_15m['fib_levels']['1.618']:.4f}

===== DATA TIMEFRAME 5 MENIT (ENTRY) =====
- Harga: ${ind_5m['current_price']:.4f}
- Trend: {ind_5m['trend']}
- RSI: {ind_5m['rsi']:.1f}
- MA20: ${ind_5m['ma20']:.4f}
- MA50: ${ind_5m['ma50']:.4f}
- MACD Line: {ind_5m['macd_line']:.6f} | Signal: {ind_5m['macd_signal']:.6f} | Histogram: {ind_5m['macd_hist']:.6f}
- Bollinger Bands: Upper ${ind_5m['bb_upper']:.4f} | Mid ${ind_5m['bb_mid']:.4f} | Lower ${ind_5m['bb_lower']:.4f}
- Support: ${ind_5m['support']:.4f} | Resistance: ${ind_5m['resistance']:.4f}
- Fibonacci Level Terdekat: {ind_5m['nearest_fib'][0]} di ${ind_5m['nearest_fib'][1]:.4f}
- Swing High: ${ind_5m['swing_high']:.4f} | Swing Low: ${ind_5m['swing_low']:.4f}
- Fib Levels: 0.382=${ind_5m['fib_levels']['0.382']:.4f} | 0.500=${ind_5m['fib_levels']['0.500']:.4f} | 0.618=${ind_5m['fib_levels']['0.618']:.4f} | 0.786=${ind_5m['fib_levels']['0.786']:.4f}
- Fib Extension: 1.272=${ind_5m['fib_levels']['1.272']:.4f} | 1.618=${ind_5m['fib_levels']['1.618']:.4f}

===== NEWS TERBARU =====
{news_text}

===== INSTRUKSI ANALISA =====
Berikan analisa SCALPING yang LENGKAP dalam format berikut:

1. 📊 KONDISI PASAR SEKARANG
   - Jelaskan kondisi dari 15m dan 5m
   - Apakah trend dan momentum selaras?

2. 📰 DAMPAK NEWS
   - Bagaimana news di atas mempengaruhi harga?
   - Apakah news mendukung atau menghambat entry?

3. 🎯 FIBONACCI ANALISA
   - Harga sekarang di level Fib mana?
   - Level support/resistance Fib terdekat?
   - Potensi bounce atau breakdown di mana?

4. 📈 SINYAL SCALPING
   - BUY / SELL / TUNGGU?
   - Jelaskan alasannya berdasarkan semua data di atas.

5. 💰 ENTRY / EXIT PLAN
   - Entry Price: (angka spesifik)
   - Take Profit 1: (level Fib extension atau resistance)
   - Take Profit 2: (level Fib extension lebih tinggi)
   - Stop Loss: (level Fib atau support)
   - Risk/Reward ratio berapa?

6. ⚠️ RISIKO
   - Apa risiko utama dari trade ini?
   - Kondisi apa yang harus diwaspadai?

Gunakan bahasa Indonesia yang JELAS dan TO THE POINT.
Jawab dengan SANGAT SPESIFIK, jangan asal-asalan angka.
"""

    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "Kamu adalah trader profesional yang ahli scalping crypto. Berikan analisa teknikal yang AKURAT, SPESIFIK, dan ACTIONABLE. Selalu kombinasikan analisa teknikal dengan sentiment news."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.4,
            max_completion_tokens=2500,
            top_p=1,
            stream=False,
            stop=None
        )
        return completion.choices[0].message.content

    except Exception as e:
        print(f"Error Groq API: {e}")
        return "⚠️ Maaf, AI analisa gagal. Coba lagi beberapa detik kemudian."

# ========== TOP GAINERS / LOSERS / VOLUME ==========
async def get_top_gainers(market_type='spot', top_n=15):
    """Ambil top gainers dari Binance"""
    try:
        exchange = exchange_spot if market_type == 'spot' else exchange_futures
        tickers = exchange.fetch_tickers()
        usdt_tickers = {
            k: v for k, v in tickers.items()
            if '/USDT' in k and v.get('percentage') is not None
        }
        sorted_tickers = sorted(usdt_tickers.items(), key=lambda x: x[1]['percentage'], reverse=True)
        return sorted_tickers[:top_n]
    except Exception as e:
        print(f"Error top gainers: {e}")
        return []

async def get_top_losers(market_type='spot', top_n=15):
    """Ambil top losers dari Binance"""
    try:
        exchange = exchange_spot if market_type == 'spot' else exchange_futures
        tickers = exchange.fetch_tickers()
        usdt_tickers = {
            k: v for k, v in tickers.items()
            if '/USDT' in k and v.get('percentage') is not None
        }
        sorted_tickers = sorted(usdt_tickers.items(), key=lambda x: x[1]['percentage'])
        return sorted_tickers[:top_n]
    except Exception as e:
        print(f"Error top losers: {e}")
        return []

async def get_top_volume(market_type='spot', top_n=15):
    """Ambil top volume dari Binance"""
    try:
        exchange = exchange_spot if market_type == 'spot' else exchange_futures
        tickers = exchange.fetch_tickers()
        usdt_tickers = {
            k: v for k, v in tickers.items()
            if '/USDT' in k and v.get('quoteVolume') is not None
        }
        sorted_tickers = sorted(usdt_tickers.items(), key=lambda x: x[1]['quoteVolume'], reverse=True)
        return sorted_tickers[:top_n]
    except Exception as e:
        print(f"Error top volume: {e}")
        return []

def format_ticker_list(ticker_list, label):
    """Format list ticker untuk Telegram"""
    if not ticker_list:
        return f"⚠️ Gagal mengambil data {label}."
    msg = f"*{label}*\n━━━━━━━━━━━━━━━━━━\n"
    for i, (symbol, data) in enumerate(ticker_list, 1):
        pct = data.get('percentage', 0) or 0
        price = data.get('last', 0) or 0
        emoji = "🟢" if pct >= 0 else "🔴"
        msg += f"{i:2d}. {emoji} *{symbol}* — ${price:.4f} ({pct:+.2f}%)\n"
    return msg

# ========== HANDLER TELEGRAM ==========
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler untuk command /start"""
    keyboard = [
        [KeyboardButton("🔥 Top Gainers 24h"), KeyboardButton("📉 Top Losers 24h")],
        [KeyboardButton("💎 Top Volume 24h")],
        [KeyboardButton("📊 All Pairs")],
        [KeyboardButton("🔄 Refresh Data")]
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)

    welcome_message = (
        "🚀 *SCALPING TRADING BOT*\n"
        "━━━━━━━━━━━━━━━━━━\n\n"
        "📊 *Fitur:*\n"
        "• Dual Timeframe: 15m + 5m\n"
        "• Indikator: RSI, MACD, Bollinger Bands\n"
        "• Fibonacci Retracement & Extension\n"
        "• News Real-time dari CryptoPanic\n"
        "• Analisa AI Groq khusus Scalping\n\n"
        "━━━━━━━━━━━━━━━━━━\n"
        "📈 *Pilih menu di bawah:*\n"
    )

    await update.message.reply_text(
        welcome_message,
        parse_mode='Markdown',
        reply_markup=reply_markup
    )

async def market_selection_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler pilihan Spot atau Futures"""
    query = update.callback_query
    await query.answer()

    market_type = query.data.split('_')[1]
    context.user_data['market_type'] = market_type

    loading_message = await query.message.reply_text("⏳ Memuat pair dari Binance...")

    pairs = get_all_pairs(market_type)

    # Keyboard 2 kolom
    keyboard = []
    row = []
    for i, pair in enumerate(pairs):
        row.append(KeyboardButton(pair))
        if (i + 1) % 2 == 0:
            keyboard.append(row)
            row = []
    if row:
        keyboard.append(row)
    keyboard.append([KeyboardButton("≡ Menu")])

    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    market_label = "SPOT" if market_type == 'spot' else "FUTURES"

    await query.edit_message_text(
        f"📊 *ALL {market_label} PAIRS*\nTotal: *{len(pairs)} pairs*\nScroll untuk lihat semua:",
        parse_mode='Markdown'
    )
    # Hapus loading message dulu
    await loading_message.delete()
    # Kirim pesan baru dengan ReplyKeyboardMarkup (tidak bisa pakai edit_text)
    await query.message.reply_text(
        f"✅ {len(pairs)} {market_label} pairs tersedia! Pilih pair di bawah 👇",
        reply_markup=reply_markup
    )

async def handle_menu_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler menu utama"""
    selected_menu = update.message.text
    market_type = context.user_data.get('market_type', 'futures')

    if selected_menu == "📊 All Pairs":
        keyboard = [[
            InlineKeyboardButton("📊 SPOT", callback_data='market_spot'),
            InlineKeyboardButton("🚀 FUTURES", callback_data='market_futures')
        ]]
        await update.message.reply_text(
            "📊 *Pilih Market Type:*",
            parse_mode='Markdown',
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    elif selected_menu == "🔥 Top Gainers 24h":
        loading = await update.message.reply_text("⏳ Mengambil Top Gainers...")
        gainers = await get_top_gainers(market_type)
        msg = format_ticker_list(gainers, "🔥 TOP GAINERS 24H")
        await loading.edit_text(msg, parse_mode='Markdown')

    elif selected_menu == "📉 Top Losers 24h":
        loading = await update.message.reply_text("⏳ Mengambil Top Losers...")
        losers = await get_top_losers(market_type)
        msg = format_ticker_list(losers, "📉 TOP LOSERS 24H")
        await loading.edit_text(msg, parse_mode='Markdown')

    elif selected_menu == "💎 Top Volume 24h":
        loading = await update.message.reply_text("⏳ Mengambil Top Volume...")
        volumes = await get_top_volume(market_type)
        msg = format_ticker_list(volumes, "💎 TOP VOLUME 24H")
        await loading.edit_text(msg, parse_mode='Markdown')

    elif selected_menu == "🔄 Refresh Data":
        # Reset cache
        PAIR_CACHE['last_update'] = None
        await update.message.reply_text("🔄 Cache di-reset. Data akan di-refresh.")
        await start(update, context)

    elif selected_menu == "≡ Menu":
        await start(update, context)

async def handle_pair_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler ketika user pilih pair atau menu"""
    selected_text = update.message.text

    # Cek menu
    menu_buttons = [
        "🔥 Top Gainers 24h", "📉 Top Losers 24h", "💎 Top Volume 24h",
        "📊 All Pairs", "🔄 Refresh Data", "≡ Menu"
    ]
    if selected_text in menu_buttons:
        await handle_menu_selection(update, context)
        return

    # Validasi format pair
    if '/USDT' not in selected_text:
        return

    selected_pair = selected_text
    market_type   = context.user_data.get('market_type', 'futures')

    # Validasi pair dari cache
    valid_pairs = get_all_pairs(market_type)
    if selected_pair not in valid_pairs:
        await update.message.reply_text(
            f"❌ Pair *{selected_pair}* tidak tersedia.\nPilih dari keyboard! 👇",
            parse_mode='Markdown'
        )
        return

    # ----- Cooldown per user -----
    user_id = update.message.user.id
    now = time.time()
    if user_id in USER_COOLDOWN and (now - USER_COOLDOWN[user_id]) < COOLDOWN_SECONDS:
        remaining = COOLDOWN_SECONDS - (now - USER_COOLDOWN[user_id])
        await update.message.reply_text(f"⏳ Tunggu dulu ya, {remaining:.0f} detik lagi.")
        return
    USER_COOLDOWN[user_id] = now

    # Loading
    market_label = "SPOT" if market_type == 'spot' else "FUTURES"
    loading_msg = await update.message.reply_text(
        f"⏳ Menganalisa *{selected_pair}* ({market_label})...\n"
        f"Ambil data 15m + 5m + News + AI...\nTunggu sebentar! 🔍",
        parse_mode='Markdown'
    )

    chart_path = None
    try:
        # 1. Ambil data dual timeframe
        ohlcv_15m = get_ohlcv_data(selected_pair, market_type, timeframe='15m', limit=100)
        ohlcv_5m  = get_ohlcv_data(selected_pair, market_type, timeframe='5m',  limit=100)

        if not ohlcv_15m or not ohlcv_5m:
            await loading_msg.edit_text("❌ Gagal ambil data dari Binance. Coba lagi!")
            return

        # 2. Hitung indikator kedua timeframe
        ind_15m = calculate_indicators(ohlcv_15m)
        ind_5m  = calculate_indicators(ohlcv_5m)

        # 3. Ambil news
        news_list      = get_crypto_news(selected_pair)
        news_for_prompt = format_news_for_prompt(news_list)
        news_for_tg    = format_news_for_telegram(news_list)

        # 4. Buat chart
        chart_path = create_dual_chart(ohlcv_5m, ohlcv_15m, ind_5m, ind_15m, selected_pair, market_type)

        # 5. AI analisa scalping
        ai_analysis = analyze_with_groq_scalping(selected_pair, ind_5m, ind_15m, news_for_prompt, market_type)

        # ===== KIRIM PESAN KE TELEGRAM =====

        # --- Kirim Chart ---
        with open(chart_path, 'rb') as chart_file:
            caption = (
                f"📊 *{selected_pair} — Scalping Chart*\n"
                f"🏷️ Market: {market_label}\n"
                f"⏰ Dual Timeframe: 15m (Trend) + 5m (Entry)\n"
                f"📐 Fibonacci + Bollinger Bands + MA"
            )
            await update.message.reply_photo(photo=chart_file, caption=caption, parse_mode='Markdown')

        # --- Kirim News ---
        await update.message.reply_text(news_for_tg, parse_mode='Markdown')

        # --- Kirim Data Teknikal Summary ---
        rsi_5m_status  = "Overbought 🔥" if ind_5m['rsi'] > 70 else ("Oversold 🧊" if ind_5m['rsi'] < 30 else "Normal ✅")
        rsi_15m_status = "Overbought 🔥" if ind_15m['rsi'] > 70 else ("Oversold 🧊" if ind_15m['rsi'] < 30 else "Normal ✅")
        macd_5m_signal = "Bullish 🟢" if ind_5m['macd_hist'] > 0 else "Bearish 🔴"
        macd_15m_signal = "Bullish 🟢" if ind_15m['macd_hist'] > 0 else "Bearish 🔴"

        summary = (
            f"📊 *DATA TEKNIKAL {selected_pair}*\n"
            f"🏷️ Market: {market_label}\n"
            f"━━━━━━━━━━━━━━━━━━\n\n"

            f"💰 *Harga Sekarang:* ${ind_5m['current_price']:.4f}\n\n"

            f"⏱️ *TIMEFRAME 15 MENIT (Trend):*\n"
            f"• Trend: {ind_15m['trend']}\n"
            f"• RSI: {ind_15m['rsi']:.1f} ({rsi_15m_status})\n"
            f"• MACD: {macd_15m_signal}\n"
            f"• BB: Upper ${ind_15m['bb_upper']:.4f} | Lower ${ind_15m['bb_lower']:.4f}\n\n"

            f"⏱️ *TIMEFRAME 5 MENIT (Entry):*\n"
            f"• Trend: {ind_5m['trend']}\n"
            f"• RSI: {ind_5m['rsi']:.1f} ({rsi_5m_status})\n"
            f"• MACD: {macd_5m_signal}\n"
            f"• BB: Upper ${ind_5m['bb_upper']:.4f} | Lower ${ind_5m['bb_lower']:.4f}\n\n"

            f"📐 *FIBONACCI:*\n"
            f"• Swing High: ${ind_5m['swing_high']:.4f}\n"
            f"• Swing Low:  ${ind_5m['swing_low']:.4f}\n"
            f"• 0.382: ${ind_5m['fib_levels']['0.382']:.4f}\n"
            f"• 0.500: ${ind_5m['fib_levels']['0.500']:.4f}\n"
            f"• 0.618: ${ind_5m['fib_levels']['0.618']:.4f} ⭐\n"
            f"• 0.786: ${ind_5m['fib_levels']['0.786']:.4f}\n"
            f"• Ext 1.272: ${ind_5m['fib_levels']['1.272']:.4f}\n"
            f"• Ext 1.618: ${ind_5m['fib_levels']['1.618']:.4f}\n"
            f"• Level Terdekat: Fib {ind_5m['nearest_fib'][0]} (${ind_5m['nearest_fib'][1]:.4f})\n\n"

            f"🔵 *Support:* ${ind_5m['support']:.4f}\n"
            f"🔴 *Resistance:* ${ind_5m['resistance']:.4f}\n"
            f"━━━━━━━━━━━━━━━━━━\n"
        )
        await update.message.reply_text(summary, parse_mode='Markdown')

        # --- Kirim AI Analisa ---
        ai_message = (
            f"🤖 *ANALISA AI — SCALPING {selected_pair}*\n"
            f"━━━━━━━━━━━━━━━━━━\n\n"
            f"{ai_analysis}\n\n"
            f"━━━━━━━━━━━━━━━━━━\n"
            f"⚠️ *Disclaimer:* Ini BUKAN rekomendasi trading!\n"
            f"Analisa ini untuk edukasi saja. Selalu lakukan riset mandiri.\n"
        )
        await update.message.reply_text(ai_message, parse_mode='Markdown')

        # Hapus loading
        await loading_msg.delete()

    except Exception as e:
        await loading_msg.edit_text(
            f"❌ Error: {str(e)}\n\n"
            f"Kemungkinan:\n"
            f"• API Binance/Groq/CryptoPanic bermasalah\n"
            f"• Pair tidak tersedia\n"
            f"• Koneksi internet bermasalah\n\n"
            f"Coba lagi ya! 🙏"
        )
        print(f"Error: {e}")
    finally:
        # Selalu hapus chart file
        if chart_path and os.path.exists(chart_path):
            os.remove(chart_path)

# ========== MAIN ==========
def main():
    """Jalankan bot"""

    # Validasi SEMUA environment variables
    if not all([TELEGRAM_TOKEN, GROQ_API_KEY, BINANCE_API_KEY, BINANCE_SECRET_KEY, CRYPTOPANIC_API_KEY]):
        print("❌ Error: Ada variabel .env yang kosong!")
        print("Pastikan .env berisi:")
        print("  TELEGRAM_BOT_TOKEN=...")
        print("  GROQ_API_KEY=...")
        print("  BINANCE_API_KEY=...")
        print("  BINANCE_SECRET_KEY=...")
        print("  CRYPTOPANIC_API_KEY=...")
        return

    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(market_selection_handler, pattern='^market_'))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_pair_selection))

    print("🤖 Scalping Bot sudah jalan!")
    print("📊 Dual Timeframe: 15m + 5m")
    print("📐 Fibonacci + BB + MACD + RSI")
    print("📰 News: CryptoPanic")
    print("🤖 AI: Groq (Llama 3.3)")
    print("💡 Tekan Ctrl+C untuk stop.")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
