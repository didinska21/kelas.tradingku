"""
CoinGecko News API Module
- FREE, no API key required
- Global trending & market data
"""

import requests
from datetime import datetime

# ========== COINGECKO NEWS API ==========
def get_crypto_news(symbol):
    """
    Ambil news dari CoinGecko - Gratis tanpa API key!
    Menggunakan: trending coins + global market data
    """
    try:
        coin = symbol.split('/')[0].lower()
        
        # 1. Get trending coins
        trending_url = "https://api.coingecko.com/api/v3/search/trending"
        response = requests.get(trending_url, timeout=5)
        
        news_list = []
        
        if response.status_code == 200:
            data = response.json()
            
            # Cari coin yang diminta di trending
            if 'coins' in data:
                for item in data['coins'][:10]:
                    coin_data = item.get('item', {})
                    symbol_match = coin_data.get('symbol', '').lower()
                    name_match = coin_data.get('name', '').lower()
                    
                    # Jika coin match dengan yang dicari
                    if coin in symbol_match or coin in name_match:
                        rank = coin_data.get('market_cap_rank', 'N/A')
                        score = coin_data.get('score', 0)
                        
                        # Determine sentiment berdasarkan trending score
                        if score >= 5:
                            sentiment = 'bullish'
                        elif score >= 3:
                            sentiment = 'neutral'
                        else:
                            sentiment = 'bearish'
                        
                        news_list.append({
                            'title': f"{coin_data.get('name')} is #trending on CoinGecko (Score: {score})",
                            'sentiment': sentiment,
                            'source': 'CoinGecko Trending',
                            'rank': rank
                        })
                
                # Jika coin tidak trending, ambil data umum trending
                if not news_list:
                    # Ambil top 3 trending untuk konteks market
                    for idx, item in enumerate(data['coins'][:3], 1):
                        coin_data = item.get('item', {})
                        news_list.append({
                            'title': f"#{idx} Trending: {coin_data.get('name')} ({coin_data.get('symbol')})",
                            'sentiment': 'neutral',
                            'source': 'CoinGecko Trending'
                        })
        
        # 2. Get global market data untuk context
        try:
            global_url = "https://api.coingecko.com/api/v3/global"
            global_response = requests.get(global_url, timeout=5)
            
            if global_response.status_code == 200:
                global_data = global_response.json().get('data', {})
                
                # Market cap change
                market_cap_change = global_data.get('market_cap_change_percentage_24h_usd', 0)
                btc_dominance = global_data.get('market_cap_percentage', {}).get('btc', 0)
                
                # Sentiment berdasarkan market cap change
                if market_cap_change > 2:
                    market_sentiment = 'bullish'
                    market_emoji = '📈'
                elif market_cap_change < -2:
                    market_sentiment = 'bearish'
                    market_emoji = '📉'
                else:
                    market_sentiment = 'neutral'
                    market_emoji = '➡️'
                
                news_list.append({
                    'title': f"{market_emoji} Global Market Cap 24h: {market_cap_change:+.2f}% | BTC Dominance: {btc_dominance:.1f}%",
                    'sentiment': market_sentiment,
                    'source': 'CoinGecko Global'
                })
        except:
            pass
        
        print(f"✅ CoinGecko: {len(news_list)} news items")
        return news_list[:5]  # Limit 5 items
        
    except requests.exceptions.RequestException as e:
        print(f"⚠️ CoinGecko API tidak dapat diakses: {type(e).__name__}")
        return []
    except Exception as e:
        print(f"⚠️ Error fetching CoinGecko news: {e}")
        return []

def format_news_for_prompt(news_list):
    """Format news untuk AI prompt"""
    if not news_list:
        return "⚠️ News tidak tersedia saat ini. Fokus pada analisa teknikal saja."
    
    formatted = ""
    for i, news in enumerate(news_list, 1):
        sentiment = news.get('sentiment', 'neutral')
        
        if sentiment == 'bullish':
            emoji = "BULLISH ✅"
        elif sentiment == 'bearish':
            emoji = "BEARISH ⚠️"
        else:
            emoji = "NETRAL ➡️"
        
        title = news.get('title', 'No title')
        formatted += f"{i}. [{emoji}] {title}\n"
    
    return formatted

def format_news_for_telegram(news_list):
    """Format news untuk Telegram"""
    if not news_list:
        return "📰 *NEWS:*\n⚠️ News tidak tersedia saat ini.\nAnalisa fokus pada teknikal."
    
    msg = "📰 *MARKET NEWS (CoinGecko):*\n"
    for i, news in enumerate(news_list, 1):
        sentiment = news.get('sentiment', 'neutral')
        
        if sentiment == 'bullish':
            emoji = "🟢 Bullish"
        elif sentiment == 'bearish':
            emoji = "🔴 Bearish"
        else:
            emoji = "🟡 Netral"
        
        title = news.get('title', 'No title')
        
        msg += f"\n{i}. {emoji}\n"
        msg += f"   {title}\n"
    
    return msg

# ========== TEST FUNCTION ==========
if __name__ == '__main__':
    """Test CoinGecko News API"""
    print("🧪 Testing CoinGecko News API\n")
    print("="*60)
    
    test_coins = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    
    for symbol in test_coins:
        print(f"\n📊 Testing {symbol}")
        print("-"*60)
        
        news = get_crypto_news(symbol)
        
        if news:
            print(f"✅ Found {len(news)} news items\n")
            print(format_news_for_telegram(news))
        else:
            print("❌ No news found")
        
        print("="*60)
    
    print("\n✅ Test completed!")
