# -*- coding: utf-8 -*-
"""
GoldTrendAlert | Liquidity Monitor Bot v1.5.10

目标：
- 输出更“可执行”的中文行动建议
- 修复缺失数据：10Y拍卖(Bid-to-Cover) + 央行购金趋势
- Telegram 使用 HTML parse_mode，实现真正加粗（<b>...</b>）

运行环境：
- Python 3.11+
- 依赖：requests

配置（优先级：环境变量 > 默认值）：
- TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
- FRED_API_KEY（若未提供，将尝试使用 DEFAULT_FRED_API_KEY）
- SOSOVALUE_API_KEY（若未提供，将尝试使用 DEFAULT_SOSO_API_KEY；也可仅用 CSV 作为回退）
- FAIR_BAND_LOW/FAIR_BAND_HIGH, PLAN_MAX_GOLD_WEIGHT, BUY_TRIGGER, TAKE_PROFIT_LEVELS
- BTC_ETF_FLOWS_CSV_URL, WGC_CSV_URL
"""

from __future__ import annotations

import csv
import html
import json
import math
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import requests

TIMEOUT = 20

# ===== Defaults (建议放到 GitHub Secrets；这里提供回退，保证“能跑”) =====
DEFAULT_FRED_API_KEY = os.getenv("DEFAULT_FRED_API_KEY", "")  # 建议使用 GitHub Secrets: FRED_API_KEY
DEFAULT_SOSO_API_KEY = os.getenv("DEFAULT_SOSO_API_KEY", "")  # 建议使用 GitHub Secrets: SOSOVALUE_API_KEY

DEFAULT_FAIR_LOW = 3600.0
DEFAULT_FAIR_HIGH = 4200.0
DEFAULT_PLAN_MAX = 18.0  # % of portfolio
DEFAULT_BUY_TRIGGER = 4100.0
DEFAULT_TAKE_PROFIT = "4600,4850,5050"

DEFAULT_BTC_ETF_CSV = "https://raw.githubusercontent.com/choosenobody/GoldTrendAlert/main/.bot_state/btc_spot_etf_flows.csv"
DEFAULT_WGC_CSV = "https://raw.githubusercontent.com/choosenobody/wgc_netbuy/main/.bot_state/wgc_netbuy.csv"

# Treasury Fiscal Data (auction results)
TREASURY_BASE = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service"
TREASURY_AUCTION_ENDPOINT = "/v1/accounting/od/auctions_query"


def env(key: str, default: str = "") -> str:
    v = os.getenv(key)
    return v if v is not None and v != "" else default


def now_cst_str() -> str:
    # CST here means China Standard Time (UTC+8)
    tz = timezone(timedelta(hours=8))
    return datetime.now(tz).strftime("%Y-%m-%d %H:%M CST")


def _get(url: str, headers: Optional[Dict[str, str]] = None, params: Optional[dict] = None) -> Optional[requests.Response]:
    try:
        r = requests.get(url, headers=headers, params=params, timeout=TIMEOUT)
        if r.status_code >= 400:
            return None
        return r
    except Exception:
        return None


def _to_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip()
        if s == "" or s.upper() == "N/A" or s == ".":
            return None
        return float(s.replace(",", ""))
    except Exception:
        return None


def fmt_money(x: Optional[float], digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "N/A"
    return f"{x:,.{digits}f}"


def fmt_pct(x: Optional[float], digits: int = 2) -> str:
    if x is None:
        return "N/A"
    return f"{x:.{digits}f}%"


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def score_linear(x: float, x0: float, x1: float, s0: float, s1: float) -> float:
    """Map x from [x0,x1] to [s0,s1] linearly, clamp outside."""
    if x0 == x1:
        return (s0 + s1) / 2
    t = (x - x0) / (x1 - x0)
    t = clamp(t, 0.0, 1.0)
    return s0 + t * (s1 - s0)


# -------------------- FRED --------------------
def fred_series_last(series_id: str, api_key: str, limit: int = 120) -> Optional[Tuple[str, float]]:
    if not api_key:
        return None
    """
    Return (date, value) for the latest non-missing observation.
    """
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": str(limit),
    }
    r = _get(url, params=params)
    if not r:
        return None
    try:
        data = r.json()
        obs = data.get("observations", [])
        for o in obs:
            v = _to_float(o.get("value"))
            if v is not None:
                return (o.get("date", ""), v)
        return None
    except Exception:
        return None


def fred_series_value_on_or_before(series_id: str, api_key: str, target_date: str) -> Optional[Tuple[str, float]]:
    if not api_key:
        return None
    """
    Find the observation on or before target_date (YYYY-MM-DD).
    We fetch a small window ending at target_date.
    """
    url = "https://api.stlouisfed.org/fred/series/observations"
    td = datetime.strptime(target_date, "%Y-%m-%d")
    start = (td - timedelta(days=45)).strftime("%Y-%m-%d")
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start,
        "observation_end": target_date,
        "sort_order": "desc",
        "limit": "200",
    }
    r = _get(url, params=params)
    if not r:
        return None
    try:
        obs = r.json().get("observations", [])
        for o in obs:
            v = _to_float(o.get("value"))
            if v is not None:
                return (o.get("date", ""), v)
        return None
    except Exception:
        return None


# -------------------- Gold spot + 波动带（ATR近似） --------------------
def stooq_xau_ohlc(last_n: int = 60) -> Optional[List[dict]]:
    """
    Stooq XAUUSD daily OHLC.
    Returns list of dicts ordered asc by date.
    """
    url = "https://stooq.com/q/d/l/"
    params = {"s": "xauusd", "i": "d"}
    r = _get(url, params=params)
    if not r:
        return None
    try:
        text = r.text.strip()
        rows = list(csv.DictReader(text.splitlines()))
        if not rows:
            return None
        rows = rows[-max(last_n, 20):]
        out = []
        for row in rows:
            dt = row.get("Date") or row.get("date") or ""
            o = _to_float(row.get("Open"))
            h = _to_float(row.get("High"))
            l = _to_float(row.get("Low"))
            c = _to_float(row.get("Close"))
            if dt and h is not None and l is not None and c is not None:
                out.append({"date": dt, "open": o, "high": h, "low": l, "close": c})
        return out if len(out) >= 20 else None
    except Exception:
        return None


def gold_price_and_vol14(fred_key: str) -> Tuple[Optional[float], Optional[float], str]:
    """
    price: try FRED GOLDAMGBD228NLBM (daily London fix). fallback to Stooq.
    vol14: use Stooq OHLC to compute 14-day ATR (Wilder's TR avg) if possible.
    """
    price = None
    src = ""
    if fred_key:
        fred = fred_series_last("GOLDAMGBD228NLBM", fred_key, limit=60)
        price = fred[1] if fred else None
        src = "FRED GOLDAMGBD228NLBM" if price is not None else ""

    ohlc = stooq_xau_ohlc(last_n=40)
    vol14 = None
    if ohlc and len(ohlc) >= 16:
        # ATR (simple avg TR over last 14 periods)
        trs = []
        prev_close = ohlc[-15]["close"]
        for bar in ohlc[-14:]:
            h, l, c = bar["high"], bar["low"], bar["close"]
            tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
            trs.append(tr)
            prev_close = c
        vol14 = sum(trs) / len(trs) if trs else None
        if price is None:
            price = ohlc[-1]["close"]
        if not src:
            src = "Stooq XAUUSD"
    else:
        # fallback price from Stooq only
        if price is None and ohlc:
            price = ohlc[-1]["close"]
            src = "Stooq XAUUSD"

    return price, vol14, src or "N/A"


# -------------------- BTC Spot ETF net flows --------------------
def sosovalue_btc_etf_weekly_flow(api_key: str) -> Optional[float]:
    """
    SoSoValue API often changes; we implement a tolerant fetch:
    - Expect JSON containing weekly net flow for BTC spot ETF.
    - If response structure mismatches, return None.
    """
    if not api_key:
        return None

    # NOTE: This endpoint may evolve. Keep logic tolerant.
    # Known pattern in earlier versions: some endpoints under "api.sosovalue.com"
    # We'll try two candidate endpoints.
    candidates = [
        ("https://api.sosovalue.com/api/v1/etf/btc/flow", {}),
        ("https://api.sosovalue.com/openapi/v1/etf/btc/flow", {}),
    ]
    headers = {"Authorization": api_key, "accept": "application/json"}

    for url, params in candidates:
        r = _get(url, headers=headers, params=params)
        if not r:
            continue
        try:
            j = r.json()
        except Exception:
            continue

        # Try a few common shapes
        # 1) {"data":{"weekNetFlow":123}} or {"data":{"weeklyNetFlow":...}}
        data = j.get("data") if isinstance(j, dict) else None
        if isinstance(data, dict):
            for k in ("weekNetFlow", "weeklyNetFlow", "week_net_flow", "weekly_net_flow"):
                v = _to_float(data.get(k))
                if v is not None:
                    return v

        # 2) {"data":[{"period":"week","netFlow":...}, ...]}
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                period = str(item.get("period", "")).lower()
                if "week" in period:
                    for k in ("netFlow", "net_flow", "netFlowUsd", "net_flow_usd", "weeklyNetFlow"):
                        v = _to_float(item.get(k))
                        if v is not None:
                            return v
        # 3) flat
        for k in ("weekNetFlow", "weeklyNetFlow"):
            v = _to_float(j.get(k))
            if v is not None:
                return v

    return None


def btc_etf_weekly_flow_from_csv(csv_url: str) -> Optional[float]:
    """
    Parse weekly net flow from a CSV.
    We accept multiple column names:
      - date / record_date
      - weekly_net_flow_usd / week_net_flow_usd / net_flow_usd
    If only daily flows exist, we sum the last 5 trading days.
    """
    r = _get(csv_url)
    if not r:
        return None
    text = r.text.strip()
    if not text:
        return None
    rows = list(csv.DictReader(text.splitlines()))
    if not rows:
        return None

    # normalize keys
    def getv(row, keys):
        for k in keys:
            if k in row and row[k] not in (None, ""):
                return row[k]
        # case-insensitive match
        low = {str(kk).lower(): kk for kk in row.keys()}
        for k in keys:
            kk = low.get(k.lower())
            if kk and row.get(kk) not in (None, ""):
                return row.get(kk)
        return None

    # try direct weekly column
    weekly_keys = ["weekly_net_flow_usd", "week_net_flow_usd", "weekly_flow_usd", "week_flow_usd"]
    last_week = None
    for row in rows[::-1]:
        v = _to_float(getv(row, weekly_keys))
        if v is not None:
            last_week = v
            break
    if last_week is not None:
        return last_week

    # else, sum last 5 available daily net flows
    daily_keys = ["net_flow_usd", "daily_net_flow_usd", "net_flow", "daily_net_flow"]
    flows = []
    for row in rows[::-1]:
        v = _to_float(getv(row, daily_keys))
        if v is not None:
            flows.append(v)
        if len(flows) >= 5:
            break
    if flows:
        return float(sum(flows))
    return None


# -------------------- Treasury auctions (10Y Note BTC) --------------------
def treasury_10y_bid_to_cover() -> Optional[Tuple[float, float, float]]:
    """
    Return (latest_btc, avg_75d, avg_2y) for 10-Year Treasury Note auctions.
    Source: Treasury Fiscal Data API auctions_query endpoint.
    """
    url = TREASURY_BASE + TREASURY_AUCTION_ENDPOINT

    today = datetime.utcnow().date()
    start_2y = (today - timedelta(days=730)).isoformat()

    # We try to filter robustly; fields vary but these are commonly present.
    # 'original_security_term' is often '10-Year'. 'security_type' is 'Note'. 'tips' is 'No'.
    params = {
        "fields": "auction_date,security_type,original_security_term,security_term,tips,bid_to_cover_ratio",
        "filter": f"auction_date:gte:{start_2y},security_type:eq:Note,tips:eq:No",
        "page[size]": "1000",
        "sort": "auction_date",
    }

    r = _get(url, params=params)
    if not r:
        return None
    try:
        data = r.json().get("data", [])
        if not isinstance(data, list) or not data:
            return None
        # filter 10Y
        records = []
        for item in data:
            if not isinstance(item, dict):
                continue
            term = (item.get("original_security_term") or item.get("security_term") or "").strip()
            if not term:
                continue
            if term.startswith("10") or "10-Year" in term or term.startswith("9-Year"):
                btc = _to_float(item.get("bid_to_cover_ratio"))
                ad = item.get("auction_date") or ""
                if btc is not None and ad:
                    records.append((ad, btc))
        if not records:
            return None
        records.sort(key=lambda x: x[0])  # date asc
        latest_date, latest_btc = records[-1]
        # compute windows
        d_latest = datetime.strptime(latest_date[:10], "%Y-%m-%d").date()
        cutoff_75 = d_latest - timedelta(days=75)

        btc_75 = [btc for d, btc in records if datetime.strptime(d[:10], "%Y-%m-%d").date() >= cutoff_75]
        btc_2y = [btc for _, btc in records]
        if not btc_75 or not btc_2y:
            return None
        avg_75 = sum(btc_75) / len(btc_75)
        avg_2y = sum(btc_2y) / len(btc_2y)
        return (latest_btc, avg_75, avg_2y)
    except Exception:
        return None


# -------------------- WGC net buy (central bank gold) --------------------
def wgc_netbuy_metrics(csv_url: str) -> Optional[Tuple[float, float, float]]:
    """
    Parse WGC net buy CSV and return:
      (last_3m_sum_tonnes, prev_3m_sum_tonnes, last_12m_sum_tonnes)
    This yields a stable "trend" measure.
    """
    r = _get(csv_url)
    if not r:
        return None
    text = r.text.strip()
    if not text:
        return None
    rows = list(csv.DictReader(text.splitlines()))
    if not rows or len(rows) < 6:
        return None

    # detect columns
    def colname(row: dict, candidates: List[str]) -> Optional[str]:
        keys = list(row.keys())
        low = {k.lower(): k for k in keys if isinstance(k, str)}
        for c in candidates:
            if c in row:
                return c
            if c.lower() in low:
                return low[c.lower()]
        return None

    date_col = colname(rows[0], ["date", "month", "period", "record_date"])
    val_col = colname(rows[0], ["net_buy_tonnes", "netbuy_tonnes", "net_buy", "net_buy_ton", "net_buy_tonnes"])
    if not date_col:
        return None
    if not val_col:
        # fallback: find first numeric-like column besides date
        for k in rows[0].keys():
            if k == date_col:
                continue
            if _to_float(rows[0].get(k)) is not None:
                val_col = k
                break
    if not val_col:
        return None

    # build monthly series
    series = []
    for row in rows:
        d = (row.get(date_col) or "").strip()
        v = _to_float(row.get(val_col))
        if not d or v is None:
            continue
        # normalize date: accept YYYY-MM or YYYY-MM-DD
        if len(d) >= 10:
            d0 = d[:10]
        elif len(d) == 7:
            d0 = d + "-01"
        else:
            continue
        series.append((d0, float(v)))

    if len(series) < 18:
        return None
    series.sort(key=lambda x: x[0])

    vals = [v for _, v in series]
    # assume monthly frequency; take last N points
    last_3 = vals[-3:]
    prev_3 = vals[-6:-3]
    last_12 = vals[-12:]
    if len(last_3) < 3 or len(prev_3) < 3 or len(last_12) < 12:
        return None
    return (sum(last_3), sum(prev_3), sum(last_12))


# -------------------- Valuation --------------------
def valuation_metrics(price: Optional[float], fair_low: float, fair_high: float) -> Tuple[Optional[float], float]:
    """
    returns (pct_vs_mid, score0_100)
    pct_vs_mid in percent (e.g. +15.8)
    """
    if price is None:
        return None, 50.0
    mid = (fair_low + fair_high) / 2.0
    pct = (price / mid - 1.0) * 100.0
    # score: within band => 50~70; above band => drop to 0; below band => rise to 100
    if price > fair_high:
        # 0% above high => 35; +25% => 0
        score = score_linear(price / fair_high - 1.0, 0.0, 0.25, 35.0, 0.0)
    elif price < fair_low:
        # 0% below low => 65; -25% => 100
        score = score_linear(fair_low / price - 1.0, 0.0, 0.25, 65.0, 100.0)
    else:
        # inside band: closer to low => higher score
        pos = (price - fair_low) / (fair_high - fair_low)
        score = 70.0 - pos * 20.0  # low=>70, high=>50
    return pct, float(clamp(score, 0.0, 100.0))


# -------------------- Action engine --------------------
@dataclass
class SignalPack:
    real_yield: Optional[float]
    real_yield_wow_pp: Optional[float]
    usd_30d_pct: Optional[float]
    etf_weekly_flow_usd: Optional[float]
    auction_latest_btc: Optional[float]
    auction_avg75: Optional[float]
    auction_avg2y: Optional[float]
    val_pct_vs_mid: Optional[float]
    val_score: float
    cb_3m_tonnes: Optional[float]
    cb_3m_mom_tonnes: Optional[float]
    cb_12m_tonnes: Optional[float]


def action_plan(price: float, vol14: Optional[float], overall: float, pack: SignalPack,
                fair_low: float, fair_high: float, plan_max: float,
                buy_trigger: float, take_profit_levels: List[float]) -> List[str]:
    """
    Produce short, executable action lines (HTML-safe later).
    """
    lines: List[str] = []

    # classify
    bias = "偏谨慎" if overall < 50 else "中性" if overall < 65 else "偏积极"
    lines.append(f"结论：<b>{bias}</b>（综合 {overall:.0f}/100）")

    # Determine state tags
    expensive = price > fair_high
    etf_out = (pack.etf_weekly_flow_usd is not None and pack.etf_weekly_flow_usd < 0)
    real_yield_up = (pack.real_yield_wow_pp is not None and pack.real_yield_wow_pp > 0)

    # Concrete allocation guidance (as % of portfolio)
    # Use "计划上限的X%" to bridge user's plan.
    plan_max = float(plan_max)
    # recommended cap within plan
    if expensive and (etf_out or real_yield_up) and overall < 50:
        # clear de-risk
        lo, hi = 0.40, 0.60
        cap_lo = plan_max * lo
        cap_hi = plan_max * hi
        lines.append(f"仓位建议：偏贵 + 资金/利率不友好 → 建议把黄金控制在<b>≤计划上限的40–60%</b>（约<b>{cap_lo:.1f}–{cap_hi:.1f}%</b>组合仓位）。")
    elif expensive and overall < 55:
        cap = plan_max * 0.70
        lines.append(f"仓位建议：估值偏贵 → 建议黄金不高于<b>计划上限的70%</b>（约<b>{cap:.1f}%</b>组合仓位）。")
    elif (not expensive) and overall >= 65 and price <= buy_trigger:
        # build-in
        build = plan_max * 0.30
        lines.append(f"仓位建议：条件偏好 + 触发回调 → 可先建<b>计划的20–30%</b>（约<b>{build*0.67:.1f}–{build:.1f}%</b>组合仓位）做起步仓。")
    else:
        lines.append(f"仓位纪律：计划黄金上限 <b>{plan_max:.1f}%</b>；<b>0仓也允许</b>（不追高）。")

    # Take profit - show next 1-2 levels above price
    tps = sorted([x for x in take_profit_levels if x > 0])
    next_tps = [x for x in tps if x >= price]
    if next_tps:
        show = next_tps[:2]
        lines.append(f"止盈位：{', '.join([f'<b>{int(x)}</b>' for x in show])}（到位分批落袋）。")
    else:
        # already above all
        lines.append(f"止盈位：{', '.join([f'<b>{int(x)}</b>' for x in tps[-2:]])}（已高位，优先锁利润）。")

    # Risk line (vol band)
    if vol14 is not None and vol14 > 0:
        base = price - 1.5 * vol14
        tight = price - 1.0 * vol14
        loose = price - 2.0 * vol14
        lines.append(
            "风控线（用“收盘价”判断）：默认 <b>1.5×波动带</b> "
            f"<b>{base:,.0f}</b>（备选 1.0× {tight:,.0f} / 2.0× {loose:,.0f}）。"
        )
        lines.append("规则：若<b>收盘价</b>跌破风控线 → 先减仓<b>50%</b>；若跌破 2.0×线 → 保留<b>0–30%</b>或清仓。")
    else:
        lines.append("风控线：波动带数据缺失（建议用近2周波动手动设“收盘破位减仓”规则）。")

    # For 0 position guidance (always helpful)
    if price > buy_trigger:
        lines.append(f"若当前为0仓：不追高；仅当回调到 ≤ <b>{buy_trigger:,.0f}</b> 再考虑分批建“观察仓”（先买计划的<b>10–20%</b>）。")

    return lines


def compute_overall(scores: List[float], weights: Optional[List[float]] = None) -> float:
    if not scores:
        return 50.0
    if not weights:
        return float(sum(scores) / len(scores))
    s = 0.0
    w = 0.0
    for sc, ww in zip(scores, weights):
        s += sc * ww
        w += ww
    return float(s / w) if w > 0 else float(sum(scores) / len(scores))


def send_telegram_html(bot_token: str, chat_id: str, html_text: str) -> bool:
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": html_text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        r = requests.post(url, json=payload, timeout=TIMEOUT)
        return r.status_code < 400
    except Exception:
        return False


def main() -> int:
    version = "v1.5.10"

    # Read config
    fred_key = env("FRED_API_KEY", DEFAULT_FRED_API_KEY)
    soso_key = env("SOSOVALUE_API_KEY", DEFAULT_SOSO_API_KEY)

    fair_low = _to_float(env("FAIR_BAND_LOW", str(DEFAULT_FAIR_LOW))) or DEFAULT_FAIR_LOW
    fair_high = _to_float(env("FAIR_BAND_HIGH", str(DEFAULT_FAIR_HIGH))) or DEFAULT_FAIR_HIGH
    plan_max = _to_float(env("PLAN_MAX_GOLD_WEIGHT", str(DEFAULT_PLAN_MAX))) or DEFAULT_PLAN_MAX
    buy_trigger = _to_float(env("BUY_TRIGGER", str(DEFAULT_BUY_TRIGGER))) or DEFAULT_BUY_TRIGGER

    tp_raw = env("TAKE_PROFIT_LEVELS", DEFAULT_TAKE_PROFIT)
    take_profit_levels = []
    for p in tp_raw.replace("，", ",").split(","):
        v = _to_float(p)
        if v is not None:
            take_profit_levels.append(float(v))
    if not take_profit_levels:
        take_profit_levels = [4600.0, 4850.0, 5050.0]

    btc_csv = env("BTC_ETF_FLOWS_CSV_URL", DEFAULT_BTC_ETF_CSV)
    wgc_csv = env("WGC_CSV_URL", DEFAULT_WGC_CSV)

    # Fetch price & vol
    price, vol14, price_src = gold_price_and_vol14(fred_key)

    # Signal 1: real yield DFII10
    ry = fred_series_last("DFII10", fred_key, limit=120)
    real_yield = ry[1] if ry else None
    # WoW change (approx 7 calendar days earlier)
    wow_date = (datetime.utcnow().date() - timedelta(days=7)).isoformat()
    ry_w = fred_series_value_on_or_before("DFII10", fred_key, wow_date)
    real_yield_wow_pp = (real_yield - ry_w[1]) if (real_yield is not None and ry_w is not None) else None

    # map score: higher real yield usually bearish for gold
    s1 = 50.0
    if real_yield is not None:
        # 0% => 65; 2% => 35; 3% => 20
        if real_yield <= 0:
            s1 = 70.0
        elif real_yield <= 2.0:
            s1 = score_linear(real_yield, 0.0, 2.0, 65.0, 35.0)
        else:
            s1 = score_linear(real_yield, 2.0, 3.5, 35.0, 15.0)
        s1 = clamp(s1, 0.0, 100.0)

    # Signal 2: broad dollar index DTWEXBGS 30d change
    last = fred_series_last("DTWEXBGS", fred_key, limit=120)
    usd_last = last[1] if last else None
    d30 = (datetime.utcnow().date() - timedelta(days=30)).isoformat()
    usd_30 = fred_series_value_on_or_before("DTWEXBGS", fred_key, d30)
    usd_30d_pct = None
    if usd_last is not None and usd_30 is not None and usd_30[1] not in (None, 0):
        usd_30d_pct = (usd_last / usd_30[1] - 1.0) * 100.0

    # score: dollar down => bullish for gold
    s2 = 50.0
    if usd_30d_pct is not None:
        # +5% => 20; 0 => 50; -5% => 80
        if usd_30d_pct >= 0:
            s2 = score_linear(usd_30d_pct, 0.0, 5.0, 50.0, 20.0)
        else:
            s2 = score_linear(usd_30d_pct, -5.0, 0.0, 80.0, 50.0)

    # Signal 3: BTC spot ETF weekly net flow
    etf_weekly_flow = sosovalue_btc_etf_weekly_flow(soso_key)
    etf_src = "SoSoValue"
    if etf_weekly_flow is None:
        etf_weekly_flow = btc_etf_weekly_flow_from_csv(btc_csv)
        etf_src = "CSV"
    # score: inflow bullish risk-on (historically sometimes bearish for gold); but user uses it as liquidity proxy.
    # We'll score: strong inflow => higher risk appetite => reduce gold => lower score.
    s3 = 50.0
    if etf_weekly_flow is not None:
        # thresholds in USD
        if etf_weekly_flow >= 2e9:
            s3 = 20.0
        elif etf_weekly_flow >= 0:
            s3 = score_linear(etf_weekly_flow, 0.0, 2e9, 55.0, 20.0)
        else:
            # outflow => more defensive => higher gold score
            s3 = score_linear(etf_weekly_flow, -2e9, 0.0, 80.0, 55.0)
        s3 = clamp(s3, 0.0, 100.0)

    # Signal 4: 10Y auction BTC
    auc = treasury_10y_bid_to_cover()
    auction_latest_btc = auction_avg75 = auction_avg2y = None
    s4 = 50.0
    if auc:
        auction_latest_btc, auction_avg75, auction_avg2y = auc
        # if demand (btc) deteriorates vs 2y avg => risk-off / rates pressure? For gold it's ambiguous.
        # We'll score: weaker demand => lower score (rates up pressure), stronger => higher score.
        delta = auction_avg75 - auction_avg2y
        # delta -0.3 => 20, 0 => 50, +0.3 => 80
        s4 = score_linear(delta, -0.3, 0.3, 20.0, 80.0)

    # Signal 5: valuation
    val_pct_vs_mid, s5 = valuation_metrics(price, fair_low, fair_high)

    # Signal 6: central bank net buy trend
    wgc = wgc_netbuy_metrics(wgc_csv)
    cb_3m = cb_prev3 = cb_12m = None
    cb_mom = None
    s6 = 50.0
    if wgc:
        cb_3m, cb_prev3, cb_12m = wgc
        cb_mom = cb_3m - cb_prev3
        # scoring: higher net buy and improving momentum => higher score
        # Use 3m sum scale: 0 -> 40, 100t -> 80 ; and momentum -50 -> 30, +50 -> 70
        base = score_linear(cb_3m, 0.0, 100.0, 40.0, 80.0)
        mom = score_linear(cb_mom, -50.0, 50.0, 30.0, 70.0)
        s6 = clamp(0.6 * base + 0.4 * mom, 0.0, 100.0)

    # overall score (weights tuned to “抓大势”: valuation + real yield + flows 更重要)
    scores = [s1, s2, s3, s4, s5, s6]
    weights = [0.22, 0.12, 0.18, 0.12, 0.26, 0.10]
    overall = compute_overall(scores, weights=weights)

    # Assemble message (HTML)
    if price is None:
        raise RuntimeError("无法获取黄金现价。请检查 FRED Key 或网络连接。")

    vol_pct = (vol14 / price * 100.0) if (vol14 is not None and price > 0) else None

    # driver lines
    drivers = []
    # 1
    ry_s = fmt_pct(real_yield, 2) if real_yield is not None else "N/A"
    ry_w_s = f"{real_yield_wow_pp:+.2f}pp" if real_yield_wow_pp is not None else "N/A"
    drivers.append(f"1 实际利率(10Y TIPS)：{ry_s}（WoW {ry_w_s}）→ {s1:.0f}")
    # 2
    usd_s = f"{usd_30d_pct:+.2f}%" if usd_30d_pct is not None else "N/A"
    drivers.append(f"2 美元指数(广义)：30d {usd_s} → {s2:.0f}")
    # 3
    etf_s = f"{etf_weekly_flow:,.0f} USD" if etf_weekly_flow is not None else "N/A"
    drivers.append(f"3 BTC现货ETF净流(周)：{etf_s}（{etf_src}）→ {s3:.0f}")
    # 4
    if auction_latest_btc is not None and auction_avg75 is not None and auction_avg2y is not None:
        delta = auction_avg75 - auction_avg2y
        drivers.append(
            f"4 10Y拍卖(BTC)：{auction_latest_btc:.2f}（75d均 {auction_avg75:.2f} vs 2y均 {auction_avg2y:.2f}，Δ {delta:+.2f}）→ {s4:.0f}"
        )
    else:
        drivers.append(f"4 10Y拍卖(BTC)：N/A → {s4:.0f}")
    # 5
    if val_pct_vs_mid is not None:
        tag = "偏贵" if price > fair_high else "偏便宜" if price < fair_low else "区间内"
        drivers.append(f"5 估值(公允 {fair_low:.0f}–{fair_high:.0f})：{tag} | 较中枢 {val_pct_vs_mid:+.1f}% → {s5:.0f}")
    else:
        drivers.append(f"5 估值：N/A → {s5:.0f}")
    # 6
    if cb_3m is not None and cb_mom is not None:
        drivers.append(f"6 央行购金(近3月净买入)：{cb_3m:.1f} 吨（较前3月 {cb_mom:+.1f}）→ {s6:.0f}")
    else:
        drivers.append(f"6 央行购金(趋势)：N/A → {s6:.0f}")

    # Action plan
    pack = SignalPack(
        real_yield=real_yield,
        real_yield_wow_pp=real_yield_wow_pp,
        usd_30d_pct=usd_30d_pct,
        etf_weekly_flow_usd=etf_weekly_flow,
        auction_latest_btc=auction_latest_btc,
        auction_avg75=auction_avg75,
        auction_avg2y=auction_avg2y,
        val_pct_vs_mid=val_pct_vs_mid,
        val_score=s5,
        cb_3m_tonnes=cb_3m,
        cb_3m_mom_tonnes=cb_mom,
        cb_12m_tonnes=cb_12m,
    )
    act_lines = action_plan(price, vol14, overall, pack, fair_low, fair_high, plan_max, buy_trigger, take_profit_levels)

    # Final message
    header = f"GoldTrendAlert | Liquidity <b>{version}</b>  {now_cst_str()}"
    line1 = f"现价：<b>${fmt_money(price, 2)}</b>  | 14日波动带：±<b>{fmt_money(vol14, 1)}</b>（{fmt_pct(vol_pct, 2)}）" if vol14 is not None else f"现价：<b>${fmt_money(price, 2)}</b>"
    body = "\n".join([
        header,
        line1,
        "",
        "<b>关键驱动（0–100）</b>",
        *drivers,
        "",
        "<b>行动（短版，可执行）</b>",
        *act_lines,
        "",
        f"来源：现价={price_src}；ETF={etf_src}；拍卖=US Treasury Fiscal Data；购金=WGC CSV",
    ])

    # Ensure HTML-safe content (except our tags)
    # Since we control content, we only escape user-provided env strings if any.
    # Here, body contains only controlled text + numbers + our tags.
    text_html = body

    # Send telegram or print
    bot = env("TELEGRAM_BOT_TOKEN")
    chat = env("TELEGRAM_CHAT_ID")
    print(html.unescape(text_html))  # show in logs

    if bot and chat:
        ok = send_telegram_html(bot, chat, text_html)
        print("Telegram sent OK." if ok else "Telegram send FAILED.", file=sys.stderr if not ok else sys.stdout)
        return 0 if ok else 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
