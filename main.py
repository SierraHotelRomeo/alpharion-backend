from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import yfinance as yf
import numpy as np
import requests
from datetime import datetime
import math

try:
    import FinanceDataReader as fdr
    FDR_AVAILABLE = True
except Exception:
    FDR_AVAILABLE = False


app = FastAPI(title="Alpharion Market Watch API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


KOREAN_NAME_MAP = {
    "삼성전자": "005930.KS",
    "SK하이닉스": "000660.KS",
    "현대차": "005380.KS",
    "기아": "000270.KS",
    "NAVER": "035420.KS",
    "네이버": "035420.KS",
    "카카오": "035720.KS",
    "LG에너지솔루션": "373220.KS",
    "LG화학": "051910.KS",
    "삼성SDI": "006400.KS",
    "POSCO홀딩스": "005490.KS",
    "셀트리온": "068270.KS",
    "삼성바이오로직스": "207940.KS",
    "두산에너빌리티": "034020.KS",
    "한화오션": "042660.KS",
    "현대로템": "064350.KS",
    "한화에어로스페이스": "012450.KS",
    "산일전기": "062040.KS",
}

KRX_CACHE = None


@app.get("/")
def root():
    return {
        "service": "Alpharion Market Watch",
        "company": "CodeGeneva Inc.",
        "status": "running"
    }


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/api/search")
def search_stock(q: str = Query("")):
    q = q.strip()

    if not q:
        return []

    results = []
    seen = set()

    # 1. 주요 국내 종목 직접 매칭
    for name, symbol in KOREAN_NAME_MAP.items():
        if q.lower() in name.lower() or q.lower() in symbol.lower():
            results.append({
                "name": name,
                "symbol": symbol,
                "market": "Korea",
                "type": "EQUITY"
            })
            seen.add(symbol)

    # 2. KRX 전체 종목명 검색
    krx_results = search_krx_by_name(q)
    for item in krx_results:
        symbol = item["symbol"]
        if symbol not in seen:
            results.append(item)
            seen.add(symbol)

    # 3. 6자리 숫자 코드 입력 시
    if q.isdigit() and len(q) == 6:
        for suffix, market in [(".KS", "Korea"), (".KQ", "Korea KOSDAQ")]:
            symbol = q + suffix
            if symbol not in seen:
                results.append({
                    "name": q,
                    "symbol": symbol,
                    "market": market,
                    "type": "EQUITY"
                })
                seen.add(symbol)

    # 4. Yahoo 전세계 검색
    yahoo_results = yahoo_search(q)
    for item in yahoo_results:
        symbol = item.get("symbol")
        if symbol and symbol not in seen:
            results.append(item)
            seen.add(symbol)

    # 5. 직접 티커 fallback
    if not results:
        guessed = normalize_symbol(q)
        results.append({
            "name": guessed,
            "symbol": guessed,
            "market": "Direct Ticker",
            "type": "UNKNOWN"
        })

    return results[:20]


@app.get("/api/stock/{symbol}")
def get_stock(symbol: str, period: str = "1y"):
    symbol = normalize_symbol(symbol)
    period = validate_period(period)

    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period, interval="1d")

        # .KS 실패 시 .KQ 재시도
        if hist.empty and symbol.endswith(".KS"):
            alt_symbol = symbol.replace(".KS", ".KQ")
            ticker = yf.Ticker(alt_symbol)
            hist = ticker.history(period=period, interval="1d")
            if not hist.empty:
                symbol = alt_symbol

        if hist.empty:
            return {"error": f"No data found for {symbol}"}

        hist = hist.dropna()

        dates = [idx.strftime("%Y-%m-%d") for idx in hist.index]
        open_prices = hist["Open"].tolist()
        high_prices = hist["High"].tolist()
        low_prices = hist["Low"].tolist()
        close_prices = hist["Close"].tolist()
        volumes = hist["Volume"].tolist()

        rsi = calculate_rsi(close_prices)

        ma5 = moving_average(close_prices, 5)
        ma20 = moving_average(close_prices, 20)
        ma60 = moving_average(close_prices, 60)
        ma120 = moving_average(close_prices, 120)

        first = close_prices[0]
        last = close_prices[-1]
        prev = close_prices[-2] if len(close_prices) >= 2 else last

        period_change = ((last - first) / first) * 100
        daily_change = ((last - prev) / prev) * 100

        forecast = ai_momentum_forecast(close_prices)
        news = news_sentiment(ticker)

        auto_signal = automatic_buy_signal(
            rsi=rsi,
            period_change=period_change,
            daily_change=daily_change,
            forecast_change=forecast["forecast_change_pct"],
            news_score=news["score"]
        )

        return {
            "symbol": symbol.upper(),
            "dates": dates,
            "open": clean_list(open_prices),
            "high": clean_list(high_prices),
            "low": clean_list(low_prices),
            "close": clean_list(close_prices),
            "volume": [int(x) if not is_bad_number(x) else 0 for x in volumes],
            "ma5": ma5,
            "ma20": ma20,
            "ma60": ma60,
            "ma120": ma120,
            "summary": {
                "last_price": round(float(last), 2),
                "daily_change": round(float(daily_change), 2),
                "period_change": round(float(period_change), 2),
                "high_price": round(float(max(high_prices)), 2),
                "low_price": round(float(min(low_prices)), 2),
                "rsi": round(float(rsi), 1),
                "signal": auto_signal["label"],
                "score": auto_signal["score"],
                "forecast_30d": round(float(forecast["forecast_change_pct"]), 2),
                "forecast_price": round(float(forecast["forecast_price"]), 2),
                "news_sentiment": news["label"],
                "news_score": news["score"],
            },
            "analysis": {
                "technical": make_technical_text(symbol, rsi, period_change, daily_change),
                "pattern": make_pattern_text(period_change, rsi),
                "lstm": forecast["text"],
                "news": news["text"],
                "auto_signal": auto_signal["text"],
                "fundamental": "현재 버전은 가격·거래량·뉴스 감성 기반 분석입니다.",
                "macro": "금리, 환율, 국채금리, 유가 등 거시지표 API를 연결하면 시장 위험도 점수화가 가능합니다.",
                "sector": "섹터별 상대수익률, 밸류에이션, 모멘텀을 연결하면 종목의 상대적 위치를 분석할 수 있습니다.",
            },
            "news": news["items"]
        }

    except Exception as e:
        return {"error": str(e)}


def get_krx_stocks():
    global KRX_CACHE

    if KRX_CACHE is not None:
        return KRX_CACHE

    stocks = []

    if not FDR_AVAILABLE:
        KRX_CACHE = stocks
        return stocks

    try:
        df = fdr.StockListing("KRX")

        for _, row in df.iterrows():
            name = str(row.get("Name", "")).strip()
            code = str(row.get("Code", "")).strip()
            market = str(row.get("Market", "")).strip()

            if not name or not code:
                continue

            if market == "KOSDAQ":
                symbol = code + ".KQ"
            else:
                symbol = code + ".KS"

            stocks.append({
                "name": name,
                "symbol": symbol,
                "market": market or "Korea",
                "type": "EQUITY"
            })

    except Exception:
        stocks = []

    KRX_CACHE = stocks
    return stocks


def search_krx_by_name(q: str):
    q = q.strip().lower()
    results = []

    if not q:
        return results

    for item in get_krx_stocks():
        name = item["name"].lower()
        symbol = item["symbol"].lower()
        pure_code = symbol.replace(".ks", "").replace(".kq", "")

        if q in name or q in symbol or q in pure_code:
            results.append(item)

        if len(results) >= 20:
            break

    return results


def yahoo_search(q: str):
    try:
        url = "https://query1.finance.yahoo.com/v1/finance/search"
        params = {
            "q": q,
            "quotesCount": 20,
            "newsCount": 0,
            "enableFuzzyQuery": "true"
        }
        headers = {"User-Agent": "Mozilla/5.0"}

        res = requests.get(url, params=params, headers=headers, timeout=8)
        data = res.json()

        results = []

        for item in data.get("quotes", []):
            symbol = item.get("symbol")
            name = item.get("shortname") or item.get("longname") or item.get("name")
            exchange = item.get("exchange") or item.get("exchDisp") or "Unknown"
            quote_type = item.get("quoteType", "")

            if symbol:
                results.append({
                    "name": name or symbol,
                    "symbol": symbol,
                    "market": exchange,
                    "type": quote_type
                })

        return results

    except Exception:
        return []


def validate_period(period: str):
    allowed = {"1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"}
    return period if period in allowed else "1y"


def normalize_symbol(value: str):
    value = value.strip()

    if value in KOREAN_NAME_MAP:
        return KOREAN_NAME_MAP[value]

    if "(" in value and ")" in value:
        start = value.find("(") + 1
        end = value.find(")")
        return value[start:end].strip()

    if value.isdigit() and len(value) == 6:
        krx_results = search_krx_by_name(value)
        if krx_results:
            return krx_results[0]["symbol"]
        return value + ".KS"

    if contains_korean(value):
        krx_results = search_krx_by_name(value)
        if krx_results:
            return krx_results[0]["symbol"]

        searched = yahoo_search(value)
        if searched:
            return searched[0]["symbol"]

    return value.upper()


def contains_korean(text: str):
    return any("가" <= ch <= "힣" for ch in text)


def is_bad_number(x):
    try:
        return x is None or math.isnan(float(x)) or math.isinf(float(x))
    except Exception:
        return True


def clean_list(values):
    cleaned = []

    for x in values:
        if is_bad_number(x):
            cleaned.append(None)
        else:
            cleaned.append(round(float(x), 2))

    return cleaned


def moving_average(values, window):
    result = []

    for i in range(len(values)):
        if i + 1 < window:
            result.append(None)
        else:
            avg = np.mean(values[i + 1 - window:i + 1])
            result.append(round(float(avg), 2))

    return result


def calculate_rsi(values, period=14):
    if len(values) < period + 1:
        return 50.0

    gains = []
    losses = []

    for i in range(1, len(values)):
        diff = values[i] - values[i - 1]
        gains.append(max(diff, 0))
        losses.append(abs(min(diff, 0)))

    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def ai_momentum_forecast(values):
    last_price = float(values[-1])
    recent = values[-30:] if len(values) >= 30 else values

    if len(recent) < 2:
        return {
            "forecast_price": last_price,
            "forecast_change_pct": 0.0,
            "text": "데이터가 부족하여 예측을 보류합니다."
        }

    momentum = ((recent[-1] - recent[0]) / recent[0]) * 100
    volatility = np.std(np.diff(recent)) / np.mean(recent) * 100

    forecast_change = momentum * 0.55 - volatility * 0.2
    forecast_price = last_price * (1 + forecast_change / 100)

    return {
        "forecast_price": forecast_price,
        "forecast_change_pct": forecast_change,
        "text": (
            f"AI 기반 30일 예측은 최근 가격 모멘텀과 변동성을 반영했습니다. "
            f"예상 가격은 약 {forecast_price:.2f}, 현재가 대비 예상 변화율은 {forecast_change:.2f}%입니다."
        )
    }


def news_sentiment(ticker):
    positive_words = [
        "beat", "growth", "strong", "surge", "record", "upgrade",
        "profit", "bullish", "gain", "ai", "demand"
    ]

    negative_words = [
        "miss", "fall", "drop", "weak", "downgrade", "loss",
        "bearish", "risk", "lawsuit", "cut", "slowdown", "concern"
    ]

    items = []

    try:
        news_list = ticker.news or []
    except Exception:
        news_list = []

    score = 0

    for n in news_list[:8]:
        title = n.get("title", "") or ""
        publisher = n.get("publisher", "") or ""
        link = n.get("link", "") or ""
        published = n.get("providerPublishTime", None)

        title_lower = title.lower()

        for w in positive_words:
            if w in title_lower:
                score += 1

        for w in negative_words:
            if w in title_lower:
                score -= 1

        date_text = ""

        if published:
            try:
                date_text = datetime.fromtimestamp(published).strftime("%Y-%m-%d")
            except Exception:
                date_text = ""

        if title:
            items.append({
                "title": title,
                "publisher": publisher,
                "link": link,
                "date": date_text
            })

    if score >= 2:
        label = "긍정"
    elif score <= -2:
        label = "부정"
    else:
        label = "중립"

    return {
        "score": score,
        "label": label,
        "items": items,
        "text": f"최근 뉴스 헤드라인 기준 감성 점수는 {score}점이며, 종합 판단은 '{label}'입니다."
    }


def automatic_buy_signal(rsi, period_change, daily_change, forecast_change, news_score):
    score = 0

    if rsi < 35:
        score += 2
    elif 35 <= rsi <= 60:
        score += 1
    elif rsi > 75:
        score -= 2

    if period_change > 8:
        score += 2
    elif period_change > 3:
        score += 1
    elif period_change < -10:
        score -= 2

    if daily_change > 0:
        score += 1
    else:
        score -= 1

    if forecast_change > 5:
        score += 2
    elif forecast_change > 0:
        score += 1
    elif forecast_change < -5:
        score -= 2

    if news_score >= 2:
        score += 1
    elif news_score <= -2:
        score -= 1

    if score >= 6:
        label = "강한 매수 관심"
    elif score >= 3:
        label = "매수 관심"
    elif score <= -3:
        label = "매수 보류"
    else:
        label = "관망"

    return {
        "score": score,
        "label": label,
        "text": (
            f"자동 매수 신호 점수는 {score}점입니다. "
            f"RSI, 기간 수익률, 단기 변동률, AI 30일 예측, 뉴스 감성 점수를 종합하여 '{label}'로 판단했습니다."
        )
    }


def make_technical_text(symbol, rsi, period_change, daily_change):
    return (
        f"{symbol.upper()}는 선택 기간 기준 {period_change:.2f}% 변동했습니다. "
        f"RSI는 {rsi:.1f}이며, 직전 거래일 대비 변동률은 {daily_change:.2f}%입니다."
    )


def make_pattern_text(period_change, rsi):
    if period_change > 10:
        return "선택 기간 동안 우상향 흐름이 나타납니다. 추세 지속형 패턴 또는 신고가 돌파 가능성을 확인해야 합니다."

    if period_change < -10:
        return "약세 흐름이 나타납니다. 지지선 이탈 여부와 거래량 증가 여부를 확인해야 합니다."

    if 45 <= rsi <= 60:
        return "강한 방향성보다는 박스권 또는 횡보 패턴 가능성이 있습니다."

    return "현재 구간은 뚜렷한 패턴보다 변동성 확인이 우선입니다."