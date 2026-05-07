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
    "팬오션": "028670.KS",
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

    for name, symbol in KOREAN_NAME_MAP.items():
        if normalize_text(q) in normalize_text(name) or normalize_text(q) in normalize_text(symbol):
            item = {
                "name": name,
                "symbol": symbol,
                "market": "Korea",
                "type": "EQUITY"
            }
            results.append(item)
            seen.add(symbol)

    for item in search_krx_by_name(q):
        symbol = item["symbol"]
        if symbol not in seen:
            results.append(item)
            seen.add(symbol)

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

    for item in yahoo_search(q):
        symbol = item.get("symbol")
        if symbol and symbol not in seen:
            results.append(item)
            seen.add(symbol)

    if not results:
        guessed = normalize_symbol(q)
        results.append({
            "name": guessed,
            "symbol": guessed,
            "market": "Direct Ticker",
            "type": "UNKNOWN"
        })

    return results[:30]


@app.get("/api/stock/{symbol}")
def get_stock(symbol: str, period: str = "1y"):
    original_input = symbol
    symbol = normalize_symbol(symbol)
    period = validate_period(period)

    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period, interval="1d")

        if hist.empty and symbol.endswith(".KS"):
            alt_symbol = symbol.replace(".KS", ".KQ")
            alt_ticker = yf.Ticker(alt_symbol)
            alt_hist = alt_ticker.history(period=period, interval="1d")
            if not alt_hist.empty:
                symbol = alt_symbol
                ticker = alt_ticker
                hist = alt_hist

        if hist.empty and symbol.endswith(".KQ"):
            alt_symbol = symbol.replace(".KQ", ".KS")
            alt_ticker = yf.Ticker(alt_symbol)
            alt_hist = alt_ticker.history(period=period, interval="1d")
            if not alt_hist.empty:
                symbol = alt_symbol
                ticker = alt_ticker
                hist = alt_hist

        if hist.empty:
            return {"error": f"No data found for {symbol}"}

        hist = hist.dropna()

        dates = [idx.strftime("%Y-%m-%d") for idx in hist.index]
        open_prices = hist["Open"].tolist()
        high_prices = hist["High"].tolist()
        low_prices = hist["Low"].tolist()
        close_prices = hist["Close"].tolist()
        volumes = hist["Volume"].tolist()

        display_name = get_display_name(symbol, original_input)
        currency = "KRW" if symbol.endswith(".KS") or symbol.endswith(".KQ") else "USD"

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
        news = safe_news_sentiment(ticker)

        auto_signal = automatic_buy_signal(
            rsi=rsi,
            period_change=period_change,
            daily_change=daily_change,
            forecast_change=forecast["forecast_change_pct"],
            news_score=news["score"]
        )

        return {
            "symbol": symbol.upper(),
            "name": display_name,
            "currency": currency,
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
                "technical": make_technical_text(display_name, rsi, period_change, daily_change),
                "pattern": make_pattern_text(period_change, rsi),
                "lstm": forecast["text"],
                "news": news["text"],
                "auto_signal": auto_signal["text"],
                "market_summary": f"{display_name}의 현재가는 {round(float(last), 2)}이며, 선택 기간 수익률은 {round(float(period_change), 2)}%입니다. 현재 신호는 '{auto_signal['label']}'입니다.",
            },
            "news": news["items"]
        }

    except Exception as e:
        return {"error": str(e)}


@app.get("/api/module/{module_id}")
def get_module(module_id: str, period: str = "6mo"):
    period = validate_period(period)

    if module_id == "market":
        return market_overview(period)
    if module_id == "fundamental":
        return market_fundamental(period)
    if module_id == "signal":
        return market_signal(period)
    if module_id == "macro":
        return macro_monitoring(period)
    if module_id == "sector_valuation":
        return sector_valuation(period)
    if module_id == "sector_momentum":
        return sector_momentum(period)
    if module_id == "market_value":
        return market_value(period)

    return {"error": "Unknown module"}


def market_overview(period):
    items = {
        "S&P 500": "^GSPC",
        "NASDAQ": "^IXIC",
        "KOSPI": "^KS11",
        "KOSDAQ": "^KQ11",
        "USD/KRW": "KRW=X",
        "WTI": "CL=F",
    }

    rows, labels, values = build_metric_rows(items, period)
    avg = safe_mean(values)
    sentiment = "긍정" if avg > 0.8 else "부정" if avg < -0.8 else "중립"

    return {
        "title": "시황",
        "subtitle": f"{period_label(period)} 기준 주요 지수·환율·원자재 시장 심리 요약",
        "cards": [
            {"label": "시장 심리", "value": sentiment},
            {"label": "평균 변동률", "value": f"{round(avg, 2)}%"},
            {"label": "관찰 지표", "value": len(rows)},
            {"label": "분석 기간", "value": period_label(period)},
        ],
        "chart_title": f"시황 변동률 - {period_label(period)}",
        "insight": f"{period_label(period)} 동안 주요 글로벌 지수, 한국 지수, 환율, 원자재 흐름을 기준으로 시장 분위기를 요약했습니다.",
        "analysis_cards": [
            {"title": "시장 심리", "text": f"평균 변동률은 {round(avg, 2)}%이며, 종합 시장 심리는 '{sentiment}'입니다."},
            {"title": "위험 요인", "text": "환율, 유가, 금리성 지표가 동시에 상승하면 위험자산 부담이 커질 수 있습니다."},
            {"title": "확인 포인트", "text": "상승 지표와 하락 지표의 비율을 확인해 단기 시장 방향성을 점검해야 합니다."},
        ],
        "rows": rows,
        "chart": {"labels": labels, "values": values, "label": "변동률 (%)"}
    }


def market_fundamental(period):
    items = {
        "S&P 500 ETF": "SPY",
        "NASDAQ ETF": "QQQ",
        "Korea ETF": "EWY",
        "US Value ETF": "VTV",
        "US Growth ETF": "VUG",
    }

    rows, labels, values = build_metric_rows(items, period)
    avg = safe_mean(values)
    status = "양호" if avg > 5 else "보통" if avg > -5 else "약화"

    return {
        "title": "펀더멘털",
        "subtitle": f"{period_label(period)} 기준 시장 ETF 기반 체력 진단",
        "cards": [
            {"label": "시장 체력", "value": status},
            {"label": "평균 수익률", "value": f"{round(avg, 2)}%"},
            {"label": "관찰 ETF", "value": len(rows)},
            {"label": "분석 기간", "value": period_label(period)},
        ],
        "chart_title": f"시장 ETF 수익률 - {period_label(period)}",
        "insight": "개별 종목이 아니라 주요 시장 ETF의 기간별 성과를 기준으로 시장의 기본 체력을 진단합니다.",
        "analysis_cards": [
            {"title": "시장 체력", "text": f"{period_label(period)} 기준 평균 수익률은 {round(avg, 2)}%이며, 시장 체력은 '{status}'로 판단됩니다."},
            {"title": "성장/가치 비교", "text": "Growth ETF와 Value ETF의 상대 흐름을 보면 시장 선호 스타일을 확인할 수 있습니다."},
            {"title": "한국시장 위치", "text": "EWY 흐름을 미국 주요 ETF와 비교하면 한국 시장의 상대 강도를 볼 수 있습니다."},
        ],
        "rows": rows,
        "chart": {"labels": labels, "values": values, "label": "Return (%)"}
    }


def market_signal(period):
    items = {
        "S&P 500": "^GSPC",
        "NASDAQ": "^IXIC",
        "KOSPI": "^KS11",
        "KOSDAQ": "^KQ11",
        "Russell 2000": "^RUT",
    }

    rows, labels, values = build_metric_rows(items, period)
    positive_count = len([v for v in values if v > 0])
    negative_count = len([v for v in values if v < 0])
    signal = "상승 우위" if positive_count > negative_count else "하락 경계" if negative_count > positive_count else "중립"

    return {
        "title": "신호",
        "subtitle": f"{period_label(period)} 기준 시장 지수 상승·하락 신호",
        "cards": [
            {"label": "시장 신호", "value": signal},
            {"label": "상승 지표", "value": positive_count},
            {"label": "하락 지표", "value": negative_count},
            {"label": "분석 기간", "value": period_label(period)},
        ],
        "chart_title": f"시장 방향성 신호 - {period_label(period)}",
        "insight": "개별 종목 신호가 아니라 주요 시장 지수의 최근 흐름을 기준으로 시장 방향성을 판단합니다.",
        "analysis_cards": [
            {"title": "상승/하락 비율", "text": f"상승 지표는 {positive_count}개, 하락 지표는 {negative_count}개이며 시장 신호는 '{signal}'입니다."},
            {"title": "시장 폭", "text": "여러 지수가 동시에 상승하면 시장 폭이 넓은 상승으로 해석할 수 있습니다."},
            {"title": "주의 구간", "text": "일부 대형 지수만 상승하고 중소형 지수가 약하면 상승 지속성을 확인해야 합니다."},
        ],
        "rows": rows,
        "chart": {"labels": labels, "values": values, "label": "Return (%)"}
    }


def macro_monitoring(period):
    items = {
        "US 10Y Yield": "^TNX",
        "Dollar Index": "DX-Y.NYB",
        "WTI Oil": "CL=F",
        "Gold": "GC=F",
        "USD/KRW": "KRW=X",
    }

    rows, labels, values = build_metric_rows(items, period)
    risk_score = len([v for v in values if v > 1])

    return {
        "title": "거시경제",
        "subtitle": f"{period_label(period)} 기준 금리·환율·원자재 모니터링",
        "cards": [
            {"label": "Macro Risk", "value": "높음" if risk_score >= 3 else "보통"},
            {"label": "관찰 지표", "value": len(rows)},
            {"label": "상승 지표", "value": risk_score},
            {"label": "분석 기간", "value": period_label(period)},
        ],
        "chart_title": f"거시경제 지표 변동률 - {period_label(period)}",
        "insight": "금리, 달러, 유가, 금, 환율을 통해 시장의 거시 위험을 점검합니다.",
        "analysis_cards": [
            {"title": "금리 분석", "text": "미국 10년물 금리 상승은 성장주 밸류에이션에 부담을 줄 수 있습니다."},
            {"title": "환율 분석", "text": "USD/KRW 상승은 외국인 수급과 수입물가 부담을 함께 확인해야 합니다."},
            {"title": "원자재 분석", "text": "유가와 금 가격은 인플레이션 및 위험회피 심리 판단에 활용됩니다."},
        ],
        "rows": rows,
        "chart": {"labels": labels, "values": values, "label": "변동률 (%)"}
    }


def sector_valuation(period):
    items = sector_items()
    rows, labels, values = build_metric_rows(items, period)
    best = labels[int(np.argmax(values))] if values else "-"

    return {
        "title": "섹터 밸류에이션",
        "subtitle": f"{period_label(period)} 기준 섹터 ETF 상대 성과",
        "cards": [
            {"label": "강세 섹터", "value": best},
            {"label": "관찰 섹터", "value": len(rows)},
            {"label": "평균 수익률", "value": f"{round(safe_mean(values), 2)}%"},
            {"label": "분석 기간", "value": period_label(period)},
        ],
        "chart_title": f"섹터 상대 성과 - {period_label(period)}",
        "insight": "섹터 ETF의 기간별 성과를 비교해 상대적으로 강한 섹터를 확인합니다.",
        "analysis_cards": [
            {"title": "강세 섹터", "text": f"{period_label(period)} 기준 가장 강한 섹터는 {best}입니다."},
            {"title": "상대 밸류", "text": "가격 성과가 강한 섹터는 이익 기대 또는 자금 유입 가능성을 함께 확인해야 합니다."},
            {"title": "분산 확인", "text": "특정 섹터만 강하면 순환매인지, 구조적 강세인지 추가 확인이 필요합니다."},
        ],
        "rows": rows,
        "chart": {"labels": labels, "values": values, "label": "Return (%)"}
    }


def sector_momentum(period):
    items = sector_items()
    rows, labels, values = build_metric_rows(items, period)
    ranked = sorted(zip(labels, values), key=lambda x: x[1], reverse=True)
    leader = ranked[0][0] if ranked else "-"

    return {
        "title": "섹터 모멘텀",
        "subtitle": f"{period_label(period)} 기준 섹터 수익률 랭킹",
        "cards": [
            {"label": "1위 섹터", "value": leader},
            {"label": "관찰 섹터", "value": len(rows)},
            {"label": "평균 모멘텀", "value": f"{round(safe_mean(values), 2)}%"},
            {"label": "분석 기간", "value": period_label(period)},
        ],
        "chart_title": f"섹터 모멘텀 - {period_label(period)}",
        "insight": "기간별 섹터 ETF 흐름을 기준으로 단기·중기 모멘텀을 측정합니다.",
        "analysis_cards": [
            {"title": "모멘텀 리더", "text": f"{period_label(period)} 기준 모멘텀 1위 섹터는 {leader}입니다."},
            {"title": "순환매 가능성", "text": "기간을 바꾸며 리더 섹터가 바뀌는지 확인하면 순환매 흐름을 볼 수 있습니다."},
            {"title": "추세 지속성", "text": "1개월과 6개월 모두 강한 섹터는 추세 지속 가능성을 더 높게 볼 수 있습니다."},
        ],
        "rows": rows,
        "chart": {"labels": labels, "values": values, "label": "Return (%)"}
    }


def market_value(period):
    items = {
        "SPY": "SPY",
        "QQQ": "QQQ",
        "DIA": "DIA",
        "IWM": "IWM",
        "EWY": "EWY",
    }

    rows, labels, values = build_metric_rows(items, period)
    avg = safe_mean(values)
    valuation = "고평가 경계" if avg > 15 else "중립" if avg > -5 else "저평가 가능성"

    return {
        "title": "시장 밸류",
        "subtitle": f"{period_label(period)} 기준 주요 ETF 고·저평가 점검",
        "cards": [
            {"label": "시장 판단", "value": valuation},
            {"label": "평균 수익률", "value": f"{round(avg, 2)}%"},
            {"label": "관찰 ETF", "value": len(rows)},
            {"label": "분석 기간", "value": period_label(period)},
        ],
        "chart_title": f"시장 밸류 점검 - {period_label(period)}",
        "insight": "주요 시장 ETF의 기간별 성과를 기준으로 시장의 고평가·저평가 가능성을 점검합니다.",
        "analysis_cards": [
            {"title": "시장 판단", "text": f"{period_label(period)} 기준 시장 판단은 '{valuation}'입니다."},
            {"title": "과열 확인", "text": "주요 ETF가 장기간 급등한 경우 단기 조정 가능성을 함께 확인해야 합니다."},
            {"title": "저평가 가능성", "text": "장기 하락 이후 회복 신호가 나타나면 저평가 반등 가능성을 점검할 수 있습니다."},
        ],
        "rows": rows,
        "chart": {"labels": labels, "values": values, "label": "Return (%)"}
    }


def sector_items():
    return {
        "Technology": "XLK",
        "Financial": "XLF",
        "Healthcare": "XLV",
        "Energy": "XLE",
        "Consumer Discretionary": "XLY",
        "Consumer Staples": "XLP",
        "Industrial": "XLI",
        "Utilities": "XLU",
    }


def build_metric_rows(items, period):
    rows = []
    labels = []
    values = []

    for name, symbol in items.items():
        metric = quick_metric(symbol, period)
        rows.append({"name": name, "value": metric["text"]})
        labels.append(name)
        values.append(metric["change"])

    return rows, labels, values


def quick_metric(symbol, period="1mo"):
    try:
        hist = yf.Ticker(symbol).history(period=period, interval="1d").dropna()
        if hist.empty:
            return {"change": 0, "text": "데이터 없음"}

        first = float(hist["Close"].iloc[0])
        last = float(hist["Close"].iloc[-1])
        change = ((last - first) / first) * 100

        return {
            "change": round(change, 2),
            "text": f"{round(last, 2)} / {round(change, 2)}%"
        }
    except Exception:
        return {"change": 0, "text": "데이터 없음"}


def get_krx_stocks():
    global KRX_CACHE

    if KRX_CACHE is not None:
        return KRX_CACHE

    stocks = []
    seen = set()

    if not FDR_AVAILABLE:
        KRX_CACHE = stocks
        return stocks

    listing_targets = ["KRX", "ETF/KR"]

    for target in listing_targets:
        try:
            df = fdr.StockListing(target)

            for _, row in df.iterrows():
                name = str(row.get("Name", "") or row.get("NameEng", "") or row.get("Symbol", "")).strip()
                code = str(row.get("Code", "") or row.get("Symbol", "")).strip()
                market = str(row.get("Market", "") or target).strip()

                if not name or not code:
                    continue

                code = code.zfill(6) if code.isdigit() and len(code) < 6 else code

                if market == "KOSDAQ":
                    symbol = code + ".KQ"
                else:
                    symbol = code + ".KS"

                key = f"{name}-{symbol}"
                if key in seen:
                    continue

                seen.add(key)

                stocks.append({
                    "name": name,
                    "symbol": symbol,
                    "market": market or "Korea",
                    "type": "ETF" if target == "ETF/KR" else "EQUITY"
                })

        except Exception:
            continue

    KRX_CACHE = stocks
    return stocks


def search_krx_by_name(q: str):
    q_norm = normalize_text(q)
    results = []

    if not q_norm:
        return results

    for item in get_krx_stocks():
        name = item["name"]
        symbol = item["symbol"]
        pure_code = symbol.replace(".KS", "").replace(".KQ", "")

        name_norm = normalize_text(name)
        symbol_norm = normalize_text(symbol)
        code_norm = normalize_text(pure_code)

        if (
            q_norm in name_norm
            or name_norm in q_norm
            or q_norm in symbol_norm
            or q_norm in code_norm
            or code_norm in q_norm
        ):
            results.append(item)

        if len(results) >= 30:
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


def period_label(period: str):
    labels = {
        "1mo": "1개월",
        "3mo": "3개월",
        "6mo": "6개월",
        "1y": "1년",
        "2y": "2년",
        "5y": "5년",
        "10y": "10년",
        "max": "전체 기간",
    }
    return labels.get(period, "1년")


def normalize_symbol(value: str):
    value = value.strip()

    if value in KOREAN_NAME_MAP:
        return KOREAN_NAME_MAP[value]

    value_norm = normalize_text(value)

    for name, code in KOREAN_NAME_MAP.items():
        if value_norm == normalize_text(name):
            return code

    if "(" in value and ")" in value:
        start = value.find("(") + 1
        end = value.find(")")
        inside = value[start:end].strip()

        krx_results = search_krx_by_name(inside)
        if krx_results:
            return krx_results[0]["symbol"]

        if "." in inside:
            return inside.upper()

        value_without_paren = value.split("(")[0].strip()
        krx_results = search_krx_by_name(value_without_paren)
        if krx_results:
            return krx_results[0]["symbol"]

        return inside.upper()

    krx_results = search_krx_by_name(value)
    if krx_results:
        return krx_results[0]["symbol"]

    if value.isdigit() and len(value) == 6:
        return value + ".KS"

    if contains_korean(value):
        searched = yahoo_search(value)
        if searched:
            return searched[0]["symbol"]

    return value.upper()


def get_display_name(symbol: str, original_input: str = ""):
    original_input = original_input.strip()

    if "(" in original_input and ")" in original_input:
        name_part = original_input.split("(")[0].strip()
        if name_part:
            return name_part

    if original_input in KOREAN_NAME_MAP:
        return original_input

    for name, code in KOREAN_NAME_MAP.items():
        if code == symbol:
            return name

    for item in get_krx_stocks():
        if item["symbol"] == symbol:
            return item["name"]

    try:
        info = yf.Ticker(symbol).info or {}
        return info.get("shortName") or info.get("longName") or symbol
    except Exception:
        return symbol


def normalize_text(text: str):
    return (
        str(text or "")
        .replace(" ", "")
        .replace("-", "")
        .replace("_", "")
        .replace("/", "")
        .replace(".", "")
        .upper()
        .strip()
    )


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


def safe_news_sentiment(ticker):
    try:
        return news_sentiment(ticker)
    except Exception:
        return {
            "score": 0,
            "label": "중립",
            "items": [],
            "text": "뉴스 데이터를 가져오지 못했습니다."
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


def make_technical_text(name, rsi, period_change, daily_change):
    return (
        f"{name}는 선택 기간 기준 {period_change:.2f}% 변동했습니다. "
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


def safe_mean(values):
    try:
        return float(np.mean(values)) if values else 0
    except Exception:
        return 0