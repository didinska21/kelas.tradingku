import os
import ccxt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
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

# ========== FUNGSI AMBIL SEMUA PAIR ==========
def get_all_pairs(market_type='spot'):
    """Ambil semua pair USDT dari Binance"""
    try:
        exchange = exchange_spot if market_type == 'spot' else exchange_futures
        markets = exchange.load_markets()
        
        # Filter hanya pair dengan quote currency USDT dan yang aktif
        usdt_pairs = []
        for symbol, market in markets.items():
            if market['quote'] == 'USDT' and market['active']:
                # Untuk spot, cek apakah market type adalah spot
                if market_type == 'spot' and market.get('spot', False):
                    usdt_pairs.append(symbol)
                # Untuk futures, cek apakah market type adalah swap/future
                elif market_type == 'futures' and (market.get('swap', False) or market.get('future', False)):
                    usdt_pairs.append(symbol)
        
        # Sort alphabetically
        usdt_pairs.sort()
        
        print(f"✅ Loaded {len(usdt_pairs)} {market_type.upper()} pairs")
        return usdt_pairs
        
    except Exception as e:
        print(f"Error loading pairs: {e}")
        # Fallback ke pairs populer
        return [
            'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT',
            'SOL/USDT', 'ADA/USDT', 'DOGE/USDT', 'AVAX/USDT'
        ]

# Load pairs saat startup (akan di-update saat pilih market)
SPOT_PAIRS = []
FUTURES_PAIRS = []

# ========== FUNGSI AMBIL DATA ==========
def get_ohlcv_data(symbol, market_type='spot', timeframe='1h', limit=100):
    """Ambil data OHLCV dari Binance"""
    try:
        exchange = exchange_spot if market_type == 'spot' else exchange_futures
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        return ohlcv
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

# ========== FUNGSI ANALISA TEKNIKAL ==========
def calculate_indicators(ohlcv_data):
    """Hitung indikator teknikal sederhana"""
    closes = np.array([x[4] for x in ohlcv_data])  # Closing prices
    highs = np.array([x[2] for x in ohlcv_data])   # High prices
    lows = np.array([x[3] for x in ohlcv_data])    # Low prices
    volumes = np.array([x[5] for x in ohlcv_data]) # Volume
    
    # Moving Averages
    ma20 = np.mean(closes[-20:]) if len(closes) >= 20 else closes[-1]
    ma50 = np.mean(closes[-50:]) if len(closes) >= 50 else closes[-1]
    
    # Support & Resistance (dari 20 candle terakhir)
    recent_lows = lows[-20:]
    recent_highs = highs[-20:]
    support = np.min(recent_lows)
    resistance = np.max(recent_highs)
    
    # Trend detection
    if closes[-1] > ma20 > ma50:
        trend = "Uptrend (Naik) 📈"
    elif closes[-1] < ma20 < ma50:
        trend = "Downtrend (Turun) 📉"
    else:
        trend = "Sideways (Datar) ↔️"
    
    # RSI sederhana (14 periods)
    deltas = np.diff(closes)
    seed = deltas[:14]
    up = seed[seed >= 0].sum() / 14
    down = -seed[seed < 0].sum() / 14
    rs = up / down if down != 0 else 0
    rsi = 100 - (100 / (1 + rs))
    
    current_price = closes[-1]
    price_change_24h = ((closes[-1] - closes[-24]) / closes[-24] * 100) if len(closes) >= 24 else 0
    
    return {
        'current_price': current_price,
        'ma20': ma20,
        'ma50': ma50,
        'support': support,
        'resistance': resistance,
        'trend': trend,
        'rsi': rsi,
        'price_change_24h': price_change_24h,
        'closes': closes,
        'highs': highs,
        'lows': lows,
        'volume': np.mean(volumes[-20:])
    }

# ========== FUNGSI BUAT CHART ==========
def create_chart(ohlcv_data, indicators, symbol, market_type):
    """Buat chart dengan garis support, resistance, dan MA"""
    timestamps = [x[0] for x in ohlcv_data]
    closes = indicators['closes']
    
    # Convert timestamps to datetime
    dates = [datetime.fromtimestamp(ts/1000) for ts in timestamps]
    
    # Create figure
    plt.figure(figsize=(14, 8))
    plt.style.use('dark_background')
    
    # Plot price line
    plt.plot(dates, closes, label='Harga', color='#00ff00', linewidth=2.5, zorder=3)
    
    # Plot Moving Averages
    if len(closes) >= 20:
        ma20_values = []
        for i in range(len(closes)):
            if i >= 19:
                ma20_values.append(np.mean(closes[i-19:i+1]))
            else:
                ma20_values.append(np.nan)
        plt.plot(dates, ma20_values, label='MA20 (Rata-rata 20)', color='#FFD700', linestyle='--', linewidth=2, zorder=2)
    
    if len(closes) >= 50:
        ma50_values = []
        for i in range(len(closes)):
            if i >= 49:
                ma50_values.append(np.mean(closes[i-49:i+1]))
            else:
                ma50_values.append(np.nan)
        plt.plot(dates, ma50_values, label='MA50 (Rata-rata 50)', color='#FF6347', linestyle='--', linewidth=2, zorder=2)
    
    # Plot Support & Resistance dengan area fill
    plt.axhline(y=indicators['support'], color='#00BFFF', linestyle='-', linewidth=2.5, label=f"🔵 Support: ${indicators['support']:.2f}", zorder=4)
    plt.axhline(y=indicators['resistance'], color='#FF1493', linestyle='-', linewidth=2.5, label=f"🔴 Resistance: ${indicators['resistance']:.2f}", zorder=4)
    
    # Fill area antara support dan resistance
    plt.fill_between(dates, indicators['support'], indicators['resistance'], alpha=0.1, color='yellow')
    
    # Annotations untuk harga saat ini
    plt.text(dates[-1], indicators['current_price'], f"  ${indicators['current_price']:.2f}", 
             color='white', fontsize=14, va='center', fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='green', alpha=0.7))
    
    # Formatting
    market_label = "SPOT" if market_type == 'spot' else "FUTURES"
    plt.title(f'📊 Analisa Chart {symbol} - {market_label}', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Waktu', fontsize=13, fontweight='bold')
    plt.ylabel('Harga (USDT)', fontsize=13, fontweight='bold')
    plt.legend(loc='upper left', fontsize=11, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle=':')
    plt.tight_layout()
    
    # Format x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m %H:%M'))
    plt.gcf().autofmt_xdate()
    
    # Save chart
    chart_path = f'chart_{symbol.replace("/", "_")}_{market_type}.png'
    plt.savefig(chart_path, dpi=120, bbox_inches='tight', facecolor='#0e1117')
    plt.close()
    
    return chart_path

# ========== FUNGSI ANALISA AI GROQ ==========
def analyze_with_groq(symbol, indicators, market_type):
    """Minta AI Groq untuk menganalisa dengan bahasa pemula"""
    
    market_label = "SPOT" if market_type == 'spot' else "FUTURES"
    
    prompt = f"""
Kamu adalah mentor trading pemula yang sangat sabar dan jelas dalam menjelaskan.

Analisa pair {symbol} ({market_label}) dengan data berikut:
- Harga Sekarang: ${indicators['current_price']:.2f}
- Perubahan 24 jam: {indicators['price_change_24h']:.2f}%
- MA20 (Moving Average 20): ${indicators['ma20']:.2f}
- MA50 (Moving Average 50): ${indicators['ma50']:.2f}
- Support (Level Bawah): ${indicators['support']:.2f}
- Resistance (Level Atas): ${indicators['resistance']:.2f}
- Trend: {indicators['trend']}
- RSI: {indicators['rsi']:.1f}

Jelaskan dengan SANGAT MUDAH untuk pemula:
1. Kondisi harga {symbol} sekarang bagaimana? (lagi naik/turun/datar)
2. Apa arti garis Support dan Resistance? Kenapa penting?
3. Apa itu Moving Average (MA20 & MA50)? Gimana cara bacanya?
4. RSI itu apa? (RSI > 70 = overbought, RSI < 30 = oversold)
5. Trend sekarang gimana? Bagus buat beli atau tunggu dulu?
6. Tips belajar untuk pemula yang lihat chart ini

Gunakan emoji, bahasa Indonesia yang gaul dan mudah dipahami anak muda. Jangan terlalu formal!
Fokus pada EDUKASI, bukan rekomendasi trading!
"""

    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "Kamu adalah mentor trading yang sangat sabar dan jelas untuk pemula. Fokus pada edukasi, bukan rekomendasi trading."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.8,
            max_completion_tokens=2000,
            top_p=1,
            stream=False,
            stop=None
        )
        
        # Ambil response dari completion
        return completion.choices[0].message.content
            
    except Exception as e:
        print(f"Error with Groq API: {e}")
        return "Maaf, terjadi error saat analisa AI. Coba lagi ya! 😊"

# ========== HANDLER TELEGRAM ==========
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler untuk command /start"""
    
    # Inline keyboard untuk pilih Spot atau Futures
    keyboard = [
        [
            InlineKeyboardButton("📊 SPOT Trading", callback_data='market_spot'),
            InlineKeyboardButton("🚀 FUTURES Trading", callback_data='market_futures')
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    welcome_message = """
🤖 *Selamat Datang di Bot Belajar Trading!*

Bot ini akan membantu kamu belajar analisa trading dengan cara yang SANGAT MUDAH dipahami! 🚀

📚 *Yang Akan Kamu Pelajari:*
✅ Cara baca chart dengan benar
✅ Apa itu Support & Resistance
✅ Cara pakai Moving Average
✅ Cara lihat trend naik/turun
✅ Indikator RSI untuk momentum

📊 *Fitur Bot:*
✅ Data real-time dari Binance
✅ Chart visual dengan garis-garis penting
✅ Penjelasan AI yang mudah dipahami
✅ Support Spot & Futures trading

⚠️ *PENTING:* 
Bot ini untuk BELAJAR saja, bukan untuk rekomendasi trading!
Selalu DYOR (Do Your Own Research) sebelum trading!

*Pilih jenis market yang mau kamu pelajari:* 👇
"""
    
    await update.message.reply_text(
        welcome_message, 
        parse_mode='Markdown', 
        reply_markup=reply_markup
    )

async def market_selection_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler untuk pilihan Spot atau Futures"""
    query = update.callback_query
    await query.answer()
    
    market_type = query.data.split('_')[1]  # 'spot' atau 'futures'
    
    # Simpan pilihan market di context
    context.user_data['market_type'] = market_type
    
    # Loading message
    loading_message = await query.message.reply_text("⏳ Memuat semua pair dari Binance...")
    
    # Ambil SEMUA pairs dari Binance
    pairs = get_all_pairs(market_type)
    
    # Buat keyboard dengan pair trading (4 kolom untuk efisiensi)
    keyboard = []
    row = []
    for i, pair in enumerate(pairs):
        # Tampilkan hanya base currency untuk menghemat space
        button_text = pair.split('/')[0]  # Contoh: BTC/USDT -> BTC
        row.append(KeyboardButton(pair))
        if (i + 1) % 4 == 0:  # 4 kolom
            keyboard.append(row)
            row = []
    if row:  # Tambahkan sisa
        keyboard.append(row)
    
    # Tambah tombol kembali dan search
    keyboard.append([KeyboardButton("🔍 Cari Pair"), KeyboardButton("🔙 Ganti Market")])
    
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    
    market_label = "SPOT" if market_type == 'spot' else "FUTURES"
    
    message = f"""
📊 *Market: {market_label} Trading*

✅ Berhasil memuat *{len(pairs)} pairs* dari Binance!

Scroll keyboard di bawah untuk lihat semua pair 👇

💡 *Tips:*
• Gunakan tombol 🔍 Cari Pair untuk cari pair tertentu
• Semua pair adalah pasangan dengan USDT

Klik pair untuk mulai analisa! 🚀
"""
    
    await query.edit_message_text(message, parse_mode='Markdown')
    await loading_message.edit_text(
        f"✅ {len(pairs)} {market_label} pairs tersedia!\nSilakan pilih dari keyboard:",
        reply_markup=reply_markup
    )


async def handle_pair_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler ketika user pilih pair trading"""
    
    selected_pair = update.message.text
    
    # Check jika user mau ganti market
    if selected_pair == "🔙 Ganti Market":
        await start(update, context)
        return
    
    # Check jika user mau search
    if selected_pair == "🔍 Cari Pair":
        await update.message.reply_text(
            "🔍 *Cara Cari Pair:*\n\n"
            "Ketik nama coin yang kamu cari, contoh:\n"
            "• BTC\n"
            "• ETH\n"
            "• DOGE\n\n"
            "Bot akan carikan pair yang cocok! 🚀",
            parse_mode='Markdown'
        )
        context.user_data['searching'] = True
        return
    
    # Ambil market type dari context
    market_type = context.user_data.get('market_type', 'spot')
    
    # Jika sedang dalam mode search
    if context.user_data.get('searching', False):
        context.user_data['searching'] = False
        pairs = get_all_pairs(market_type)
        
        # Cari pair yang match
        search_term = selected_pair.upper().replace('/USDT', '')
        matched_pairs = [p for p in pairs if search_term in p]
        
        if not matched_pairs:
            await update.message.reply_text(
                f"❌ Tidak ada pair dengan nama '{selected_pair}'\n"
                f"Coba kata kunci lain atau pilih dari keyboard! 👇"
            )
            return
        elif len(matched_pairs) == 1:
            selected_pair = matched_pairs[0]
        else:
            # Tampilkan pilihan
            result = f"🔍 Ditemukan {len(matched_pairs)} pair:\n\n"
            for p in matched_pairs[:10]:  # Max 10
                result += f"• {p}\n"
            result += "\nKlik salah satu dari keyboard! 👇"
            await update.message.reply_text(result)
            return
    
    # Validasi format pair
    if '/USDT' not in selected_pair:
        return
    
    # Validasi pair berdasarkan market
    valid_pairs = get_all_pairs(market_type)
    
    if selected_pair not in valid_pairs:
        await update.message.reply_text(
            f"❌ Pair *{selected_pair}* tidak tersedia di market ini.\n"
            f"Gunakan 🔍 Cari Pair atau pilih dari keyboard! 👇",
            parse_mode='Markdown'
        )
        return
    
    # Kirim loading message
    market_label = "SPOT" if market_type == 'spot' else "FUTURES"
    loading_msg = await update.message.reply_text(
        f"⏳ Sedang menganalisa {selected_pair} ({market_label})...\n"
        f"Mohon tunggu 10-15 detik ya! 🔍📊"
    )
    
    try:
        # 1. Ambil data dari Binance
        ohlcv_data = get_ohlcv_data(selected_pair, market_type, timeframe='1h', limit=100)
        if not ohlcv_data:
            await loading_msg.edit_text("❌ Gagal mengambil data dari Binance. Coba lagi ya!")
            return
        
        # 2. Hitung indikator
        indicators = calculate_indicators(ohlcv_data)
        
        # 3. Buat chart
        chart_path = create_chart(ohlcv_data, indicators, selected_pair, market_type)
        
        # 4. Analisa dengan AI Groq
        ai_analysis = analyze_with_groq(selected_pair, indicators, market_type)
        
        # 5. Kirim chart
        with open(chart_path, 'rb') as chart_file:
            caption = f"📊 *Chart Analisa {selected_pair}*\n🏷️ Market: {market_label}\n⏰ Timeframe: 1 Jam"
            await update.message.reply_photo(
                photo=chart_file, 
                caption=caption,
                parse_mode='Markdown'
            )
        
        # 6. Kirim ringkasan data
        change_emoji = "🟢" if indicators['price_change_24h'] >= 0 else "🔴"
        rsi_status = "Overbought 🔥" if indicators['rsi'] > 70 else "Oversold 🧊" if indicators['rsi'] < 30 else "Normal ✅"
        
        summary = f"""
📊 *DATA TEKNIKAL {selected_pair}*
🏷️ Market: {market_label}

💰 *Harga Sekarang:* ${indicators['current_price']:.2f}
{change_emoji} *Perubahan 24h:* {indicators['price_change_24h']:.2f}%

📈 *Indikator:*
• Trend: {indicators['trend']}
• RSI: {indicators['rsi']:.1f} ({rsi_status})
• MA20: ${indicators['ma20']:.2f}
• MA50: ${indicators['ma50']:.2f}

🔵 *Support:* ${indicators['support']:.2f}
🔴 *Resistance:* ${indicators['resistance']:.2f}

━━━━━━━━━━━━━━━━━━
"""
        await update.message.reply_text(summary, parse_mode='Markdown')
        
        # 7. Kirim analisa AI
        ai_message = f"""
🤖 *PENJELASAN AI UNTUK PEMULA*

{ai_analysis}

━━━━━━━━━━━━━━━━━━

💡 *PANDUAN BACA CHART:*

🔵 *Garis Biru (Support)*
Ini "lantai" harga. Kalau harga turun ke sini, biasanya susah turun lagi karena banyak yang beli (demand tinggi).

🔴 *Garis Pink (Resistance)*  
Ini "plafon" harga. Kalau harga naik ke sini, biasanya susah naik lagi karena banyak yang jual (supply tinggi).

💛 *Garis Kuning (MA20)*
Rata-rata harga 20 jam terakhir. Kalau harga di atas garis = momentum kuat! Kalau di bawah = momentum lemah.

❤️ *Garis Merah (MA50)*
Rata-rata harga 50 jam terakhir. Untuk lihat trend jangka menengah.

🟢 *Garis Hijau*
Harga aktual yang bergerak real-time.

━━━━━━━━━━━━━━━━━━

📚 *Tips Belajar:*
1. Jangan langsung trading! Amati dulu 1-2 minggu
2. Pahami setiap garis dan artinya
3. Catat prediksi vs hasil aktual
4. Belajar dari kesalahan analisa

⚠️ *Disclaimer:* Ini BUKAN rekomendasi trading! Hanya untuk belajar analisa teknikal.

Mau analisa pair lain? 
• Gunakan 🔍 Cari Pair untuk cari cepat
• Atau scroll keyboard di bawah! 👇
"""
        
        await update.message.reply_text(ai_message, parse_mode='Markdown')
        
        # Hapus loading message
        await loading_msg.delete()
        
        # Hapus file chart
        if os.path.exists(chart_path):
            os.remove(chart_path)
            
    except Exception as e:
        await loading_msg.edit_text(
            f"❌ Terjadi error: {str(e)}\n\n"
            f"Kemungkinan:\n"
            f"• API Binance/Groq bermasalah\n"
            f"• Pair tidak tersedia\n"
            f"• Koneksi internet bermasalah\n\n"
            f"Coba lagi ya! 🙏"
        )
        print(f"Error: {e}")

# ========== MAIN ==========
def main():
    """Jalankan bot"""
    
    # Validasi environment variables
    if not all([TELEGRAM_TOKEN, GROQ_API_KEY]):
        print("❌ Error: TELEGRAM_BOT_TOKEN atau GROQ_API_KEY tidak ditemukan di .env!")
        print("Pastikan file .env sudah dibuat dengan format:")
        print("TELEGRAM_BOT_TOKEN=your_token_here")
        print("GROQ_API_KEY=your_groq_key_here")
        return
    
    # Buat aplikasi
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Tambahkan handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(market_selection_handler, pattern='^market_'))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_pair_selection))
    
    # Jalankan bot
    print("🤖 Bot Trading Edukasi sudah jalan!")
    print("📊 Support: Spot & Futures")
    print("💡 Tekan Ctrl+C untuk stop.")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()
