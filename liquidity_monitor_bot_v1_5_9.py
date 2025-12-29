#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GoldTrendAlert | Liquidity Monitor Bot v1.5.9

v1.5.9 改进点（对应你的需求）：
1) 修复缺失数据：
   - 用 FRED API 拉 DFII10 与 DTWEXBGS（支持你给的 FRED key；仍建议放 Secrets）
   - ETF 净流：默认用你给的公开 CSV；CSV 异常时，尝试 SoSoValue API 兜底
   - 央行购金：默认用你给的 WGC CSV（做 12M YoY 趋势信号）
2) 用更易懂的“14日平均波动”替代 ATR（本质仍是 ATR 类指标，但不再用 ATR 术语）
3) “仓位纪律”补齐：0仓/低仓/超上限 的执行建议；支持可选 CURRENT_GOLD_WEIGHT(%)
4) 版本号 v1.5.9（文件名与消息头一致）
5) 容错增强：列名自适应、缺数据不中断、消息更短只保留关键结论与动作

注意：
- 公共仓库最佳实践：把 key 放 GitHub Secrets（脚本仍做了 env 优先 + fallback，方便你立刻跑通）
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import math
import os
import statistics
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests


# -----------------------
# Config / Env
# -----------------------

DEFAULT_FRED_API_KEY_FALLBACK = "46eccf9075cdf430632c2d47d01185ce"
DEFAULT_SOSOVALUE_API_KEY_FALLBACK = "SOSO-78fd0f7109494e209dad0fe7e1f1ac12"

DEFAULT_BTC_ETF_FLOWS_CSV_URL = (
    "https://raw.githubusercontent.com/choosenobody/GoldTrendAlert/main/.bot_state/btc_spot_etf_flows.csv"
)
DEFAULT_WGC_CSV_URL = (
    "https://raw.githubusercontent.com/choosenobody/wgc_netbuy/main/.bot_state/wgc_netbuy.csv"
)

DEFAULT_FAIR_BAND_LOW = 3600.0
DEFAULT_FAIR_BAND_HIGH = 4200.0
DEFAULT_PLAN_MAX_GOLD_WEIGHT = 18.0  # %
DEFAULT_BUY_TRIGGER = 4100.0
DEFAULT_BUY_BANDS = "3960-3920,3920-3850,3850-3780"
DEFAULT_TAKE_PROFIT_LEVELS = "4600,4850,5050"
DEFAULT_ETF_LONG_D = 34
DEFAULT_ETF_SHORT_D = 13


def _env(key: str, default: str = "") -> str:
    v = os.getenv(key, "")
    return v.strip() if isinstance(v, str) else default


def _env_float(key: str, default: float) -> float:
    s = _env(key, "")
    if not s:
        return default
    try:
        return float(s)
    except Exception:
        return default


def _env_int(key: str, default: int) -> int:
    s = _env(key, "")
    if not s:
        return default
    try:
        return int(float(s))
    except Exception:
        return default


def _now_cn() -> dt.datetime:
    return dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).astimezone(dt.timezone(dt.timedelta(hours=8)))


# -----------------------
# HTTP helpers
# -----------------------

UA = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0 Safari/537.36"
)

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": UA})


def http_get(url: str, timeout: int = 25) -> str:
    r = SESSION.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text


def http_post_json(url: str, payload: Dict[str, Any], headers: Optional[Dict[str, str]] = None, timeout: int = 25) -> Dict[str, Any]:
    h = {"User-Agent": UA, "Content-Type": "application/json"}
    if headers:
        h.update(headers)
    r = SESSION.post(url, data=json.dumps(payload), headers=h, timeout=timeout)
    r.raise_for_status()
    return r.json()


# -----------------------
# Data fetchers
# -----------------------

@dataclass
class PriceBar:
    date: dt.date
    open: float
    high: float
    low: float
    close: float


def fetch_gold_ohlc_stooq(days: int = 220) -> List[PriceBar]:
    """Stooq 日线：https://stooq.com/q/d/l/?s=xauusd&i=d"""
    url = "https://stooq.com/q/d/l/?s=xauusd&i=d"
    text = http_get(url, timeout=25)
    rows: List[PriceBar] = []
    reader = csv.DictReader(text.splitlines())
    for row in reader:
        try:
            d = dt.datetime.strptime(row["Date"], "%Y-%m-%d").date()
            o = float(row["Open"])
            h = float(row["High"])
            l = float(row["Low"])
            c = float(row["Close"])
            rows.append(PriceBar(d, o, h, l, c))
        except Exception:
            continue
    rows.sort(key=lambda x: x.date)
    return rows[-days:] if days and len(rows) > days else rows


def fetch_gold_price_yahoo() -> Optional[float]:
    """Yahoo 实时 XAUUSD=X（失败则回退到 Stooq 收盘价）"""
    url = "https://query1.finance.yahoo.com/v7/finance/quote?symbols=XAUUSD%3DX"
    try:
        data = SESSION.get(url, timeout=20).json()
        res = (data.get("quoteResponse") or {}).get("result") or []
        if res:
            p = res[0].get("regularMarketPrice")
            return float(p) if p is not None else None
    except Exception:
        return None
    return None


def fred_series_observations(series_id: str, api_key: str, limit: int = 800) -> List[Tuple[dt.date, Optional[float]]]:
    """FRED 观测：/fred/series/observations"""
    api_key = api_key.strip()
    url = (
        "https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}&api_key={api_key}&file_type=json&sort_order=asc&limit={int(limit)}"
    )
    js = json.loads(http_get(url, timeout=25))
    obs = js.get("observations") or []
    out: List[Tuple[dt.date, Optional[float]]] = []
    for o in obs:
        ds = o.get("date")
        vs = o.get("value")
        try:
            d = dt.datetime.strptime(ds, "%Y-%m-%d").date()
        except Exception:
            continue
        if vs in (None, ".", ""):
            out.append((d, None))
            continue
        try:
            out.append((d, float(vs)))
        except Exception:
            out.append((d, None))
    return out


def latest_non_null(series: List[Tuple[dt.date, Optional[float]]]) -> Optional[Tuple[dt.date, float]]:
    for d, v in reversed(series):
        if v is not None and math.isfinite(v):
            return (d, float(v))
    return None


def nearest_on_or_before(series: List[Tuple[dt.date, Optional[float]]], target: dt.date) -> Optional[Tuple[dt.date, float]]:
    best: Optional[Tuple[dt.date, float]] = None
    for d, v in series:
        if d > target:
            break
        if v is None or not math.isfinite(v):
            continue
        best = (d, float(v))
    return best


def pct_change(a: float, b: float) -> Optional[float]:
    if b == 0:
        return None
    return (a / b - 1.0) * 100.0


def parse_csv_flexible(text: str) -> List[Dict[str, str]]:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []
    reader = csv.DictReader(lines)
    return [dict(r) for r in reader]


def detect_col(row: Dict[str, str], candidates: Sequence[str]) -> Optional[str]:
    keys = {k.lower(): k for k in row.keys()}
    for c in candidates:
        if c.lower() in keys:
            return keys[c.lower()]
    return None


def parse_date_any(s: str) -> Optional[dt.date]:
    s = (s or "").strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return dt.datetime.strptime(s[:19], fmt).date()
        except Exception:
            continue
    return None


def float_or_none(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        s = str(x).strip().replace(",", "")
        if s in ("", ".", "null", "None"):
            return None
        return float(s)
    except Exception:
        return None


def compute_ema(values: List[float], span: int) -> List[float]:
    if not values:
        return []
    span = max(1, int(span))
    alpha = 2.0 / (span + 1.0)
    out = [values[0]]
    for v in values[1:]:
        out.append(alpha * v + (1 - alpha) * out[-1])
    return out


# -----------------------
# Signals
# -----------------------

@dataclass
class Signal:
    name: str
    value_str: str
    score: float


def score_real_yield_dfii10(api_key: str) -> Signal:
    series = fred_series_observations("DFII10", api_key, limit=800)
    latest = latest_non_null(series)
    if not latest:
        return Signal("实际利率 DFII10", "N/A", 50.0)
    d0, v0 = latest
    back = nearest_on_or_before(series, d0 - dt.timedelta(days=7))
    wow = None if not back else (v0 - back[1])

    # 实际利率越低越利多黄金（经验映射）
    if v0 <= 0.5:
        score = 85.0
    elif v0 >= 2.5:
        score = 20.0
    else:
        score = 85.0 + (v0 - 0.5) * (20.0 - 85.0) / (2.5 - 0.5)

    value_str = f"{v0:.2f}%（WoW {wow:+.2f}pp）" if wow is not None else f"{v0:.2f}%（WoW N/A）"
    return Signal("实际利率 DFII10", value_str, float(score))


def score_broad_dollar_dtwexbgs(api_key: str) -> Signal:
    series = fred_series_observations("DTWEXBGS", api_key, limit=900)
    latest = latest_non_null(series)
    if not latest:
        return Signal("广义美元 DTWEXBGS", "N/A", 50.0)
    d0, v0 = latest
    back30 = nearest_on_or_before(series, d0 - dt.timedelta(days=30))
    chg30 = None if not back30 else pct_change(v0, back30[1])

    # 30日美元越弱越利多黄金
    if chg30 is None:
        score = 50.0
        value_str = "30d N/A"
    elif chg30 <= -2.0:
        score = 80.0
        value_str = f"30d {chg30:+.2f}%"
    elif chg30 >= 2.0:
        score = 20.0
        value_str = f"30d {chg30:+.2f}%"
    else:
        score = 80.0 + (chg30 + 2.0) * (20.0 - 80.0) / 4.0
        value_str = f"30d {chg30:+.2f}%"

    return Signal("广义美元 DTWEXBGS", value_str, float(score))


def fetch_btc_etf_flows_from_csv(url: str) -> Optional[List[Tuple[dt.date, float]]]:
    try:
        text = http_get(url, timeout=30)
    except Exception:
        return None
    rows = parse_csv_flexible(text)
    if not rows:
        return None

    date_col = None
    val_col = None
    for r in rows[:3]:
        date_col = date_col or detect_col(r, ["date", "day", "dt"])
        val_col = val_col or detect_col(r, ["totalNetInflow", "totalnetinflow", "total_net_inflow", "net_inflow", "netflow", "flow"])

    if not date_col:
        for k in rows[0].keys():
            if "date" in k.lower():
                date_col = k
                break

    if not val_col and date_col:
        sample = rows[0]
        for k in sample.keys():
            if k == date_col:
                continue
            if float_or_none(sample.get(k)) is not None:
                val_col = k
                break

    if not date_col or not val_col:
        return None

    out: List[Tuple[dt.date, float]] = []
    for r in rows:
        d = parse_date_any(r.get(date_col, ""))
        v = float_or_none(r.get(val_col, ""))
        if d and v is not None and math.isfinite(v):
            out.append((d, float(v)))
    out.sort(key=lambda x: x[0])
    return out if out else None


def fetch_btc_etf_flows_from_sosovalue(api_key: str) -> Optional[List[Tuple[dt.date, float]]]:
    if not api_key:
        return None
    headers = {"x-soso-api-key": api_key}
    bases = ["https://open.sosovalue.xyz", "https://openapi.sosovalue.com"]
    path = "/openapi/v1/etf/us-btc-spot/historicalInflowChart"
    for base in bases:
        try:
            js = http_post_json(base + path, {}, headers=headers, timeout=30)
            if (js.get("code") == 0) and isinstance(js.get("data"), list):
                out: List[Tuple[dt.date, float]] = []
                for it in js["data"]:
                    d = parse_date_any(str(it.get("date", "")))
                    v = float_or_none(it.get("totalNetInflow"))
                    if d and v is not None:
                        out.append((d, float(v)))
                out.sort(key=lambda x: x[0])
                if out:
                    return out
        except Exception:
            continue
    return None


def score_btc_etf_flow(short_d: int, long_d: int) -> Signal:
    url = _env("BTC_ETF_FLOWS_CSV_URL") or DEFAULT_BTC_ETF_FLOWS_CSV_URL
    api_key = _env("SOSOVALUE_API_KEY") or DEFAULT_SOSOVALUE_API_KEY_FALLBACK

    series = fetch_btc_etf_flows_from_csv(url)
    src = "CSV"
    if not series:
        series = fetch_btc_etf_flows_from_sosovalue(api_key)
        src = "SoSoValue" if series else "N/A"

    if not series or len(series) < 10:
        return Signal("BTC现货ETF净流", "N/A", 50.0)

    vals = [v for _, v in series]
    week = sum(vals[-5:]) if len(vals) >= 5 else sum(vals)

    ema_s = compute_ema(vals, short_d)
    ema_l = compute_ema(vals, long_d)
    slope = ema_l[-1] - ema_l[-6] if len(ema_l) >= 6 else (ema_l[-1] - ema_l[0])

    base = 50.0 + max(-25.0, min(25.0, week / 2_000_000_000.0 * 25.0))
    trend = (10.0 if ema_s[-1] > ema_l[-1] else -10.0) + (5.0 if slope > 0 else (-5.0 if slope < 0 else 0.0))
    score = max(0.0, min(100.0, base + trend))

    value_str = f"周净流 {week:,.0f} USD（{src}）"
    return Signal("BTC现货ETF净流", value_str, float(score))


def fetch_wgc_series_from_csv(url: str) -> Optional[List[Tuple[dt.date, float]]]:
    try:
        text = http_get(url, timeout=30)
    except Exception:
        return None
    rows = parse_csv_flexible(text)
    if not rows:
        return None

    sample = rows[0]
    date_col = None
    for k in sample.keys():
        if any(x in k.lower() for x in ("date", "month", "time")):
            date_col = k
            break
    if not date_col:
        date_col = list(sample.keys())[0]

    val_col = None
    for k in sample.keys():
        if k == date_col:
            continue
        if float_or_none(sample.get(k)) is not None:
            val_col = k
            break
    if not val_col and len(sample.keys()) >= 2:
        val_col = list(sample.keys())[1]

    out: List[Tuple[dt.date, float]] = []
    for r in rows:
        d = parse_date_any(r.get(date_col, ""))
        v = float_or_none(r.get(val_col, ""))
        if d and v is not None and math.isfinite(v):
            out.append((d, float(v)))
    out.sort(key=lambda x: x[0])
    return out if out else None


def score_wgc_cb_buy_yoy() -> Signal:
    url = _env("WGC_CSV_URL") or DEFAULT_WGC_CSV_URL
    series = fetch_wgc_series_from_csv(url)
    if not series or len(series) < 24:
        return Signal("央行购金(趋势)", "N/A", 50.0)

    vals = [v for _, v in series]
    sum_12 = sum(vals[-12:])
    sum_prev_12 = sum(vals[-24:-12])
    yoy = pct_change(sum_12, sum_prev_12)

    if yoy is None:
        score = 50.0
        value_str = "YoY N/A"
    elif yoy >= 20.0:
        score = 75.0
        value_str = f"12M YoY {yoy:+.1f}%"
    elif yoy <= -20.0:
        score = 30.0
        value_str = f"12M YoY {yoy:+.1f}%"
    else:
        score = 30.0 + (yoy + 20.0) * (75.0 - 30.0) / 40.0
        value_str = f"12M YoY {yoy:+.1f}%"

    return Signal("央行购金(趋势)", value_str, float(score))


# -----------------------
# Volatility / Stops (14日平均波动)
# -----------------------

@dataclass
class VolInfo:
    price: float
    avg_move14: Optional[float]      # USD
    avg_move14_pct: Optional[float]  # %
    stops: Dict[str, float]


def compute_avg_move14_from_ohlc(bars: List[PriceBar]) -> Optional[float]:
    if len(bars) < 15:
        return None
    trs: List[float] = []
    prev_close = bars[0].close
    for b in bars[1:]:
        tr = max(b.high - b.low, abs(b.high - prev_close), abs(b.low - prev_close))
        trs.append(tr)
        prev_close = b.close
    if len(trs) < 14:
        return None
    return statistics.mean(trs[-14:])


def vol_and_stops() -> VolInfo:
    price = fetch_gold_price_yahoo()
    bars: List[PriceBar] = []
    try:
        bars = fetch_gold_ohlc_stooq(days=220)
    except Exception:
        bars = []

    if price is None and bars:
        price = bars[-1].close

    if price is None or not math.isfinite(price):
        return VolInfo(float("nan"), None, None, {})

    avg_move14 = compute_avg_move14_from_ohlc(bars) if bars else None
    if avg_move14 is None and bars and len(bars) >= 15:
        closes = [b.close for b in bars]
        diffs = [abs(closes[i] - closes[i-1]) for i in range(1, len(closes))]
        avg_move14 = statistics.mean(diffs[-14:]) if len(diffs) >= 14 else None

    if avg_move14 is None or not math.isfinite(avg_move14):
        return VolInfo(float(price), None, None, {})

    avg_pct = (avg_move14 / price) * 100.0 if price else None
    stops = {
        "1.0x": price - 1.0 * avg_move14,
        "1.5x": price - 1.5 * avg_move14,
        "2.0x": price - 2.0 * avg_move14,
    }
    return VolInfo(float(price), float(avg_move14), float(avg_pct) if avg_pct is not None else None, stops)


# -----------------------
# Valuation
# -----------------------

def score_valuation(price: float, fair_low: float, fair_high: float) -> Signal:
    if not (math.isfinite(price) and fair_low > 0 and fair_high > fair_low):
        return Signal("估值(公允区间)", "N/A", 50.0)

    mid = (fair_low + fair_high) / 2.0
    if price <= fair_low:
        disc = (fair_low - price) / fair_low
        score = min(100.0, 85.0 + disc * 75.0)
        label = "偏便宜"
    elif price >= fair_high:
        prem = (price - fair_high) / fair_high
        score = max(0.0, 40.0 - prem * 200.0)  # +20% => 0
        label = "偏贵"
    else:
        score = 70.0 + (price - fair_low) * (40.0 - 70.0) / (fair_high - fair_low)
        label = "公允区间"

    dev_mid = (price / mid - 1.0) * 100.0
    value_str = f"{label} | 相对中枢 {dev_mid:+.1f}%"
    return Signal("估值(公允区间)", value_str, float(score))


# -----------------------
# Auction quality (keep neutral unless you wire datasource)
# -----------------------

def score_ust_auction_btc() -> Signal:
    raw = _env("UST_AUCTIONS_DELTA", "")
    if raw and raw.strip().startswith("{"):
        try:
            js = json.loads(raw)
            last = float(js.get("last"))
            avg75 = float(js.get("avg75d"))
            avg2y = float(js.get("avg2y"))
            delta = avg75 - avg2y
            score = 50.0 + max(-10.0, min(10.0, delta * 50.0))
            return Signal("10Y拍卖BTC", f"{last:.2f}（Δ{delta:+.2f}）", float(score))
        except Exception:
            pass
    return Signal("10Y拍卖BTC", "N/A", 50.0)


# -----------------------
# Composite & message
# -----------------------

def clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def overall_score(signals: List[Signal]) -> float:
    w = {
        "实际利率 DFII10": 0.25,
        "广义美元 DTWEXBGS": 0.20,
        "BTC现货ETF净流": 0.25,
        "10Y拍卖BTC": 0.10,
        "估值(公允区间)": 0.15,
        "央行购金(趋势)": 0.05,
    }
    s = 0.0
    tw = 0.0
    for sig in signals:
        ww = w.get(sig.name, 0.0)
        s += ww * sig.score
        tw += ww
    return clamp(s / tw) if tw > 0 else 50.0


def stance_from_score(score: float) -> str:
    if score >= 70:
        return "偏多"
    if score >= 55:
        return "中性偏多"
    if score >= 45:
        return "中性"
    if score >= 30:
        return "偏谨慎"
    return "防守"


def parse_ranges(s: str) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        a, b = (p.strip() for p in part.split("-", 1))
        aa = float_or_none(a)
        bb = float_or_none(b)
        if aa is None or bb is None:
            continue
        out.append((float(aa), float(bb)))
    return out


def build_action_lines(score: float, price: float, vol: VolInfo, fair_low: float, fair_high: float, plan_max: float) -> List[str]:
    stance = stance_from_score(score)
    buy_trigger = _env_float("BUY_TRIGGER", DEFAULT_BUY_TRIGGER)
    buy_bands = parse_ranges(_env("BUY_BANDS") or DEFAULT_BUY_BANDS)
    take_profit = [float(x.strip()) for x in (_env("TAKE_PROFIT_LEVELS") or DEFAULT_TAKE_PROFIT_LEVELS).split(",") if float_or_none(x) is not None]
    cur_w = float_or_none(_env("CURRENT_GOLD_WEIGHT", ""))  # optional

    lines: List[str] = []
    if stance in ("防守", "偏谨慎"):
        lines.append(f"- **{stance}**：先风控/锁利润，避免追高加仓。")
    elif stance == "中性":
        lines.append("- **中性**：不追高不恐慌，等回调/等信号再动作。")
    else:
        lines.append(f"- **{stance}**：允许分批加仓，但只做“回调买入”。")

    if math.isfinite(price):
        if price >= fair_high:
            lines.append(f"- 估值偏贵（>公允上沿 {fair_high:,.0f}）：更偏向‘不追高/先锁利润’。")
        elif price <= fair_low:
            lines.append(f"- 估值偏便宜（<公允下沿 {fair_low:,.0f}）：更偏向‘分批布局’。")
        else:
            lines.append("- 估值在公允区间：以‘回调分批’为主。")

    # Take profit: nearest above
    if take_profit and math.isfinite(price):
        tps = sorted(take_profit)
        above = [x for x in tps if x > price]
        if above:
            lines.append(f"- 首个止盈位：**{above[0]:,.0f}**（到位分批落袋）。")
        else:
            lines.append("- 已高于全部止盈位：建议分批落袋，并上移风控线。")

    # Stops from 14d avg move
    if vol.avg_move14 and vol.stops:
        s15 = vol.stops.get("1.5x")
        s10 = vol.stops.get("1.0x")
        s20 = vol.stops.get("2.0x")
        if s15 and s10 and s20:
            lines.append(f"- 风控线（默认 1.5×14日平均波动）：**{s15:,.0f}**  | 备选：1.0× {s10:,.0f} / 2.0× {s20:,.0f}")
            lines.append("  规则：若**收盘价**跌破你的风控线 → 按计划减仓 **50–100%**。")

    # Position discipline (0仓/超上限)
    if cur_w is None:
        lines.append(f"- 仓位纪律：计划黄金上限 **{plan_max:.1f}%**；**0仓也允许**（不追高）。")
        lines.append(f"  - 若当前为0仓：仅当回调到 ≤ **{buy_trigger:,.0f}** 再分批建仓（先建计划的20–30%）。")
    else:
        lines.append(f"- 仓位纪律：当前约 **{cur_w:.1f}%**，计划上限 **{plan_max:.1f}%**。")
        if cur_w > plan_max + 0.1:
            lines.append("  - 已超上限：优先减到 ≤ 上限（先锁利润/降回撤）。")
        elif cur_w < 0.1:
            lines.append(f"  - 当前≈0仓：不追高；回调到 ≤ **{buy_trigger:,.0f}** 再开始分批建仓（先20–30%计划仓）。")
        else:
            lines.append("  - 未超上限：按回调规则与信号强弱决定加/减仓。")

    if buy_bands:
        bands_txt = "，".join([f"{a:,.0f}-{b:,.0f}" for a, b in buy_bands[:3]])
        lines.append(f"- 加仓规则：只在回调到 ≤ **{buy_trigger:,.0f}** 再考虑；分批区间：{bands_txt}。")

    return lines


def build_message() -> str:
    fred_key = _env("FRED_API_KEY") or DEFAULT_FRED_API_KEY_FALLBACK
    short_d = _env_int("ETF_SHORT_D", DEFAULT_ETF_SHORT_D)
    long_d = _env_int("ETF_LONG_D", DEFAULT_ETF_LONG_D)

    fair_low = _env_float("FAIR_BAND_LOW", DEFAULT_FAIR_BAND_LOW)
    fair_high = _env_float("FAIR_BAND_HIGH", DEFAULT_FAIR_BAND_HIGH)
    plan_max = _env_float("PLAN_MAX_GOLD_WEIGHT", DEFAULT_PLAN_MAX_GOLD_WEIGHT)

    vol = vol_and_stops()
    price = vol.price

    sigs: List[Signal] = []
    sigs.append(score_real_yield_dfii10(fred_key))
    sigs.append(score_broad_dollar_dtwexbgs(fred_key))
    sigs.append(score_btc_etf_flow(short_d, long_d))
    sigs.append(score_ust_auction_btc())
    sigs.append(score_valuation(price, fair_low, fair_high))
    sigs.append(score_wgc_cb_buy_yoy())

    total = overall_score(sigs)
    stance = stance_from_score(total)

    ts = _now_cn().strftime("%Y-%m-%d %H:%M CST")

    if vol.avg_move14 is not None and vol.avg_move14_pct is not None:
        vol_str = f"14日平均波动：{vol.avg_move14:,.1f}（{vol.avg_move14_pct:.2f}%）"
    else:
        vol_str = "14日平均波动：N/A"

    lines: List[str] = []
    lines.append(f"GoldTrendAlert | Liquidity v1.5.9  {ts}")
    if math.isfinite(price):
        lines.append(f"现价：${price:,.2f}  | {vol_str}")
    else:
        lines.append(f"现价：N/A  | {vol_str}")
    lines.append(f"综合：{int(round(total))}/100（{stance}）")
    lines.append("")
    lines.append("关键驱动（0–100）：")
    for i, s in enumerate(sigs, start=1):
        lines.append(f"{i} {s.name}：{s.value_str} → {int(round(s.score))}")
    lines.append("")
    lines.append("行动（短版，中文执行）：")
    lines.extend(build_action_lines(total, price, vol, fair_low, fair_high, plan_max))
    return "\n".join(lines).strip() + "\n"


# -----------------------
# Telegram
# -----------------------

def send_telegram(text: str) -> None:
    token = _env("TELEGRAM_BOT_TOKEN")
    chat_id = _env("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        print("WARN: TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID 未配置，仅打印消息。", file=sys.stderr)
        print(text)
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "disable_web_page_preview": True}
    r = SESSION.post(url, data=payload, timeout=20)
    if not r.ok:
        raise RuntimeError(f"Telegram 发送失败: {r.status_code} {r.text[:200]}")
    print("Telegram sent OK.")


def main() -> int:
    try:
        msg = build_message()
        print(msg)
        send_telegram(msg)
        return 0
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
