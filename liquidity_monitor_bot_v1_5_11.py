#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GoldTrendAlert | Liquidity Monitor Bot v1.5.11

本版改动（针对你的反馈）：
1) 修复 #3 BTC现货ETF净流(周) 的 N/A：
   - CSV 解析更“自适应”：支持 totalNetInflow / total_net_inflow / net_flow_usd / flow 等多种列名；
   - 如果找不到列名，则自动选择“除日期列外的第一个数值列”；
   - 若 CSV 仍失败，则回退 SoSoValue（使用更可靠的 open.sosovalue.xyz / openapi.sosovalue.com 的历史流入接口）。
2) 修复 #4 10Y拍卖(BTC) 的 N/A：
   - Treasury Fiscal Data API 仅做最小过滤（security_type=Note），其余在本地筛选“10-Year 且非TIPS”，更耐字段变化。
3) 修复 #6 央行购金(趋势) 的 N/A：
   - WGC CSV 同样自适应：自动识别日期列 + 首个数值列（不再强依赖 net_buy_tonnes 这种列名）。
4) “风控线”表达更容易决策：
   - 明确 3 条线（紧/标准/宽）；
   - 只用“收盘价”判定 → 下一交易日执行；
   - 给出清晰 if/then。
5) Telegram 真加粗：parse_mode=HTML，使用 <b>..</b>。

依赖：requests
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone, date
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests


# -------------------- Defaults --------------------
TIMEOUT = 25

DEFAULT_FRED_API_KEY = "46eccf9075cdf430632c2d47d01185ce"
DEFAULT_SOSO_API_KEY = "SOSO-78fd0f7109494e209dad0fe7e1f1ac12"

DEFAULT_FAIR_LOW = 3600.0
DEFAULT_FAIR_HIGH = 4200.0
DEFAULT_PLAN_MAX = 18.0  # % of portfolio
DEFAULT_BUY_TRIGGER = 4100.0
DEFAULT_BUY_BANDS = "3960-3920,3920-3850,3850-3780"
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


def _post_json(url: str, headers: Optional[Dict[str, str]] = None, payload: Optional[dict] = None) -> Optional[requests.Response]:
    try:
        r = requests.post(url, headers=headers, json=(payload or {}), timeout=TIMEOUT)
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
        s = str(x).strip().replace(",", "")
        if s in ("", "N/A", "NA", "null", "None", "."):
            return None
        return float(s)
    except Exception:
        return None


def clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, x))


def score_linear(x: float, x0: float, x1: float, s0: float, s1: float) -> float:
    if x1 == x0:
        return (s0 + s1) / 2
    t = (x - x0) / (x1 - x0)
    return s0 + t * (s1 - s0)


# -------------------- Price + Vol (14日波动带) --------------------
def stooq_xau_last_price() -> Optional[float]:
    # Stooq CSV: date,open,high,low,close
    url = "https://stooq.com/q/d/l/?s=xauusd&i=d"
    r = _get(url)
    if not r:
        return None
    lines = r.text.strip().splitlines()
    if len(lines) < 3:
        return None
    last = lines[-1].split(",")
    try:
        return float(last[4])
    except Exception:
        return None


def stooq_xau_ohlc(last_n: int = 40) -> Optional[List[Dict[str, Any]]]:
    url = "https://stooq.com/q/d/l/?s=xauusd&i=d"
    r = _get(url)
    if not r:
        return None
    rows = list(csv.DictReader(r.text.strip().splitlines()))
    if not rows:
        return None
    rows = rows[-last_n:] if last_n and len(rows) > last_n else rows
    out = []
    for row in rows:
        try:
            out.append(
                {
                    "date": row["Date"],
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                }
            )
        except Exception:
            continue
    return out if out else None


def compute_vol14_band(price: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
    """
    返回：(vol14_usd, vol14_pct)
    这里的 vol14 是基于 TR 的 14 日平均（接近 ATR，但对用户表述为“14日波动带”）。
    """
    ohlc = stooq_xau_ohlc(last_n=40)
    if not ohlc or len(ohlc) < 16:
        return None, None

    # 计算 14 日 TR 平均
    trs = []
    prev_close = ohlc[-15]["close"]
    for bar in ohlc[-14:]:
        h, l, c = bar["high"], bar["low"], bar["close"]
        tr = max(h - l, abs(h - prev_close), abs(l - prev_close))
        trs.append(tr)
        prev_close = c
    if not trs:
        return None, None

    vol14 = sum(trs) / len(trs)
    p = price if price is not None else ohlc[-1]["close"]
    vol_pct = (vol14 / p * 100.0) if p else None
    return float(vol14), (float(vol_pct) if vol_pct is not None else None)


# -------------------- FRED helpers --------------------
def fred_series(series_id: str, api_key: str, limit: int = 800) -> List[Tuple[date, Optional[float]]]:
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "asc",
        "limit": str(limit),
    }
    r = _get(url, params=params)
    if not r:
        return []
    try:
        j = r.json()
    except Exception:
        return []
    obs = j.get("observations") or []
    out: List[Tuple[date, Optional[float]]] = []
    for o in obs:
        ds = o.get("date")
        vs = o.get("value")
        try:
            d0 = datetime.strptime(ds, "%Y-%m-%d").date()
        except Exception:
            continue
        v0 = None if vs in (None, ".", "") else _to_float(vs)
        out.append((d0, v0))
    return out


def latest_non_null(series: List[Tuple[date, Optional[float]]]) -> Optional[Tuple[date, float]]:
    for d0, v0 in reversed(series):
        if v0 is not None and math.isfinite(v0):
            return d0, float(v0)
    return None


def nearest_on_or_before(series: List[Tuple[date, Optional[float]]], target: date) -> Optional[Tuple[date, float]]:
    best: Optional[Tuple[date, float]] = None
    for d0, v0 in series:
        if d0 > target:
            break
        if v0 is None or not math.isfinite(v0):
            continue
        best = (d0, float(v0))
    return best


# -------------------- BTC Spot ETF net flows --------------------
def sosovalue_btc_etf_weekly_flow(api_key: str) -> Optional[float]:
    """
    更可靠的 SoSoValue 抓取方式：
    - 使用 historicalInflowChart（返回每日 totalNetInflow），我们求最近 5 条之和 = 周净流（近似）
    """
    if not api_key:
        return None

    bases = ["https://open.sosovalue.xyz", "https://openapi.sosovalue.com"]
    path = "/openapi/v1/etf/us-btc-spot/historicalInflowChart"

    # 两种常见 header 方案都试一下
    header_variants = [
        {"x-soso-api-key": api_key},
        {"Authorization": api_key},
    ]

    for base in bases:
        for hdr in header_variants:
            r = _post_json(base + path, headers=hdr, payload={})
            if not r:
                continue
            try:
                j = r.json()
            except Exception:
                continue
            data = j.get("data")
            if not isinstance(data, list) or len(data) < 5:
                continue

            vals: List[float] = []
            for it in data[-10:][::-1]:
                if not isinstance(it, dict):
                    continue
                v = _to_float(it.get("totalNetInflow"))
                if v is not None:
                    vals.append(float(v))
                if len(vals) >= 5:
                    break
            if len(vals) >= 3:
                return float(sum(vals))
    return None


def _detect_date_col(fieldnames: Sequence[str]) -> Optional[str]:
    if not fieldnames:
        return None
    # 优先包含 date 的列
    for k in fieldnames:
        if "date" in k.lower():
            return k
    for k in fieldnames:
        if any(x in k.lower() for x in ("day", "dt", "time")):
            return k
    return fieldnames[0]


def _detect_value_col(fieldnames: Sequence[str]) -> Optional[str]:
    """
    ETF CSV 的列名变化很大：尽量宽松匹配；
    若都不匹配，后续会用“首个数值列”兜底。
    """
    if not fieldnames:
        return None
    candidates = [
        "weekly_net_flow_usd", "week_net_flow_usd", "weekly_flow_usd", "week_flow_usd",
        "net_flow_usd", "daily_net_flow_usd", "daily_net_flow", "net_flow",
        "totalNetInflow", "total_net_inflow", "totalnetinflow",
        "totalNetInflowUsd", "total_net_inflow_usd",
        "flow", "inflow", "netInflow", "net_inflow",
    ]
    low = {k.lower(): k for k in fieldnames}
    for c in candidates:
        if c in fieldnames:
            return c
        if c.lower() in low:
            return low[c.lower()]
    return None


def _parse_date_any(s: str) -> Optional[date]:
    s = (s or "").strip()
    if not s:
        return None
    fmts = ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S")
    for fmt in fmts:
        try:
            return datetime.strptime(s[:19], fmt).date()
        except Exception:
            continue
    return None


def btc_etf_weekly_flow_from_csv(csv_url: str) -> Optional[float]:
    """
    读取 CSV，优先：
    - 若存在“weekly/周净流”列：直接取最新一条
    - 否则：用“日净流”列求最近 5 条之和
    - 列名不确定：自动识别日期列 + 数值列
    """
    r = _get(csv_url)
    if not r:
        return None
    text = r.text.strip()
    if not text:
        return None

    reader = csv.DictReader(text.splitlines())
    rows = list(reader)
    if not rows:
        return None

    fieldnames = reader.fieldnames or list(rows[0].keys())
    date_col = _detect_date_col(fieldnames)
    val_col = _detect_value_col(fieldnames)

    # 1) 尝试直接取“周净流列”
    weekly_keys = ["weekly_net_flow_usd", "week_net_flow_usd", "weekly_flow_usd", "week_flow_usd"]
    low = {k.lower(): k for k in fieldnames}
    wk_cols = [low.get(k.lower()) for k in weekly_keys if low.get(k.lower())]
    for col in wk_cols:
        for row in rows[::-1]:
            v = _to_float(row.get(col))
            if v is not None:
                return float(v)

    # 2) 若 val_col 存在，先按它走；否则用“首个数值列”兜底
    def first_numeric_col(sample_row: dict) -> Optional[str]:
        if not isinstance(sample_row, dict):
            return None
        for k, vv in sample_row.items():
            if date_col and k == date_col:
                continue
            if _to_float(vv) is not None:
                return k
        return None

    if not val_col:
        val_col = first_numeric_col(rows[0])
        if not val_col:
            return None

    # 3) 用 val_col 求最近 5 条（按日期排序更稳）
    series: List[Tuple[date, float]] = []
    for row in rows:
        d0 = _parse_date_any((row.get(date_col) if date_col else "") or "")
        v0 = _to_float(row.get(val_col))
        if d0 and v0 is not None:
            series.append((d0, float(v0)))
    if not series:
        return None
    series.sort(key=lambda x: x[0])

    # 近 5 条求和
    last_vals = [v for _, v in series[-5:]]
    return float(sum(last_vals)) if last_vals else None


# -------------------- Treasury auctions (10Y Note BTC) --------------------
def treasury_10y_bid_to_cover() -> Optional[Tuple[float, float, float]]:
    """
    Return (latest_btc, avg_75d, avg_2y) for 10-Year Treasury Note auctions.
    做“最小过滤”，再本地筛选：
    - security_type=Note
    - term 包含 10 + year（排除 20/30）
    - tips != Yes
    """
    try:
        url = TREASURY_BASE + TREASURY_AUCTION_ENDPOINT
        today = datetime.utcnow().date()
        start_2y = (today - timedelta(days=730)).isoformat()

        params = {
            "fields": "auction_date,security_type,original_security_term,security_term,tips,bid_to_cover_ratio",
            "filter": f"auction_date:gte:{start_2y},security_type:eq:Note",
            "page[size]": "1000",
            "sort": "auction_date",
        }
        r = _get(url, params=params)
        if not r:
            return None
        j = r.json()
        data = j.get("data") or []
        if not isinstance(data, list) or not data:
            return None

        def is_10y_note(row: dict) -> bool:
            term = str(row.get("original_security_term") or row.get("security_term") or "").lower()
            tips = str(row.get("tips") or "").lower()
            if "yes" in tips or tips in ("y", "true", "1"):
                return False
            # term must contain "year" and "10"
            if "year" not in term:
                return False
            if not re.search(r"\b10\b", term) and "10-" not in term and "10year" not in term and "10-year" not in term:
                return False
            # exclude 20/30 if present
            if re.search(r"\b20\b", term) or re.search(r"\b30\b", term):
                return False
            return True

        series: List[Tuple[date, float]] = []
        for row in data:
            if not isinstance(row, dict):
                continue
            if not is_10y_note(row):
                continue
            d = str(row.get("auction_date") or "")[:10]
            btc = _to_float(row.get("bid_to_cover_ratio"))
            if not d or btc is None:
                continue
            try:
                d0 = datetime.strptime(d, "%Y-%m-%d").date()
            except Exception:
                continue
            series.append((d0, float(btc)))

        if not series:
            return None
        series.sort(key=lambda x: x[0])
        latest_btc = series[-1][1]

        cutoff_75 = today - timedelta(days=75)
        btc_75 = [v for d0, v in series if d0 >= cutoff_75]
        btc_2y = [v for _, v in series]

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
    自适应列名：日期列 + 首个数值列。
    """
    r = _get(csv_url)
    if not r:
        return None
    text = r.text.strip()
    if not text:
        return None

    reader = csv.DictReader(text.splitlines())
    rows = list(reader)
    if not rows or len(rows) < 6:
        return None

    fieldnames = reader.fieldnames or list(rows[0].keys())

    # date col
    date_col = None
    for k in fieldnames:
        kl = k.lower()
        if "date" in kl or "month" in kl or "period" in kl or "time" in kl:
            date_col = k
            break
    if not date_col:
        date_col = fieldnames[0]

    # value col: try known names then fallback to first numeric
    cand = ["net_buy_tonnes", "netbuy_tonnes", "net_buy", "netbuy", "tonnes", "tons", "ton"]
    low = {k.lower(): k for k in fieldnames}
    val_col = None
    for c in cand:
        if c.lower() in low:
            val_col = low[c.lower()]
            break
    if not val_col:
        for k in fieldnames:
            if k == date_col:
                continue
            if _to_float(rows[0].get(k)) is not None:
                val_col = k
                break
    if not val_col:
        return None

    series: List[Tuple[str, float]] = []
    for row in rows:
        d = (row.get(date_col) or "").strip()
        v = _to_float(row.get(val_col))
        if not d or v is None:
            continue
        # accept YYYY-MM or YYYY-MM-DD
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
    pct_vs_mid: 相对中枢偏离（%）
    """
    if price is None:
        return None, 50.0
    mid = (fair_low + fair_high) / 2.0
    pct = (price / mid - 1.0) * 100.0

    # score: below fair_low => higher; above fair_high => lower
    if price <= fair_low:
        score = clamp(75.0 + (fair_low - price) / fair_low * 100.0, 0.0, 100.0)
    elif price >= fair_high:
        # 价格高于上沿越多，越低分；+20% 直接 0
        premium = (price / fair_high - 1.0) * 100.0
        score = clamp(40.0 - premium * 2.0, 0.0, 100.0)
    else:
        # fair band 内：从 70 线性降到 40
        score = score_linear(price, fair_low, fair_high, 70.0, 40.0)

    return float(pct), float(score)


# -------------------- Telegram --------------------
def telegram_send_html(text_html: str) -> bool:
    token = env("TELEGRAM_BOT_TOKEN", "")
    chat_id = env("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        print("WARN: TELEGRAM_BOT_TOKEN/CHAT_ID 未配置，仅打印：")
        print(text_html)
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text_html,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        r = requests.post(url, data=payload, timeout=TIMEOUT)
        return r.status_code < 400
    except Exception:
        return False


# -------------------- Message builder --------------------
@dataclass
class Pack:
    price: Optional[float]
    vol14: Optional[float]
    vol14_pct: Optional[float]
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float
    dfii10: Optional[float]
    dfii10_wow: Optional[float]
    usd_30d_pct: Optional[float]
    etf_weekly_flow: Optional[float]
    etf_src: str
    auc: Optional[Tuple[float, float, float]]
    cb: Optional[Tuple[float, float, float]]


def stance(overall: float) -> str:
    if overall >= 70:
        return "偏多"
    if overall >= 55:
        return "中性偏多"
    if overall >= 45:
        return "中性"
    if overall >= 30:
        return "偏谨慎"
    return "防守"


def parse_take_profit(s: str) -> List[float]:
    out = []
    for part in (s or "").split(","):
        v = _to_float(part.strip())
        if v is not None:
            out.append(float(v))
    return sorted(list(set(out)))


def format_int(n: Optional[float]) -> str:
    if n is None or not math.isfinite(n):
        return "N/A"
    return f"{n:,.0f}"


def build_actions(pack: Pack, fair_low: float, fair_high: float, plan_max: float) -> List[str]:
    price = pack.price
    vol = pack.vol14
    overall = (pack.s1 * 0.25 + pack.s2 * 0.20 + pack.s3 * 0.25 + pack.s4 * 0.10 + pack.s5 * 0.15 + pack.s6 * 0.05) / 1.0
    overall = clamp(overall)

    st = stance(overall)

    # 估值偏贵判断
    expensive = (price is not None and price >= fair_high)

    # 仓位建议更“抓趋势”：给明确上限（以计划上限为基准）
    cap_ratio = 1.0
    if expensive and overall < 45:
        cap_ratio = 0.70
    elif expensive and overall < 55:
        cap_ratio = 0.85
    elif (not expensive) and overall >= 60:
        cap_ratio = 1.00
    elif (not expensive) and overall < 40:
        cap_ratio = 0.60

    cap_weight = plan_max * cap_ratio

    # 风控线（紧/标准/宽）：用“收盘价”触发 → 次日执行
    tight = std = loose = None
    if price is not None and vol is not None:
        tight = price - 1.0 * vol
        std = price - 1.5 * vol
        loose = price - 2.0 * vol

    buy_trigger = _to_float(env("BUY_TRIGGER", str(DEFAULT_BUY_TRIGGER))) or DEFAULT_BUY_TRIGGER

    lines: List[str] = []
    lines.append(f"<b>结论：{st}（综合 {int(round(overall))}/100）</b>")

    if expensive:
        lines.append(f"<b>仓位建议：</b>估值偏贵 → 建议黄金不高于计划上限的<b>{int(round(cap_ratio*100))}%</b>（约<b>{cap_weight:.1f}%</b>组合仓位）")
    else:
        lines.append(f"<b>仓位建议：</b>未明显偏贵 → 计划上限内顺势分批（上限<b>{plan_max:.1f}%</b>）")

    # 止盈
    tps = parse_take_profit(env("TAKE_PROFIT_LEVELS", DEFAULT_TAKE_PROFIT))
    if price is not None and tps:
        above = [x for x in tps if x > price]
        if above:
            first_tp = above[0]
            lines.append(f"<b>止盈：</b>{', '.join(str(int(x)) for x in tps[:3])}（先盯<b>{int(first_tp)}</b>，到位分批落袋）")
        else:
            lines.append("<b>止盈：</b>已高于全部止盈位 → 建议分批落袋 + 上移风控线")

    # 风控（更易懂的 if/then）
    if tight is not None and std is not None and loose is not None:
        lines.append(
            f"<b>风控（只看“收盘价”，次日执行）：</b>"
            f"紧<b>{int(tight):,}</b> / 标准<b>{int(std):,}</b> / 宽<b>{int(loose):,}</b>"
        )
        lines.append(
            f"- 若收盘价 < <b>{int(std):,}</b>（标准线）→ 次日先减仓<b>50%</b>"
        )
        lines.append(
            f"- 若收盘价 < <b>{int(loose):,}</b>（宽线）→ 保留<b>0–30%</b>或清仓（按你的风险偏好）"
        )

    lines.append(f"<b>0仓策略：</b>不追高；仅当回调到 ≤ <b>{int(buy_trigger):,}</b> 再建“观察仓”（先买计划的<b>10–20%</b>）")
    return lines


def build_message(pack: Pack, fair_low: float, fair_high: float) -> str:
    version = "v1.5.11"
    ts = now_cst_str()

    price = pack.price
    p_str = f"${price:,.2f}" if price is not None else "N/A"
    if pack.vol14 is not None and pack.vol14_pct is not None:
        vol_str = f"±{pack.vol14:,.1f}（{pack.vol14_pct:.2f}%）"
    else:
        vol_str = "N/A"

    # signals text
    s1 = f"{pack.dfii10:.2f}%" if pack.dfii10 is not None else "N/A"
    wow = f"{pack.dfii10_wow:+.2f}pp" if pack.dfii10_wow is not None else "N/A"
    usd = f"{pack.usd_30d_pct:+.2f}%" if pack.usd_30d_pct is not None else "N/A"
    etf = f"{format_int(pack.etf_weekly_flow)} USD" if pack.etf_weekly_flow is not None else "N/A"

    auc_str = "N/A"
    if pack.auc:
        last_btc, avg75, avg2y = pack.auc
        delta = avg75 - avg2y
        auc_str = f"{last_btc:.2f}（75d均{avg75:.2f} vs 2y均{avg2y:.2f}，Δ{delta:+.2f}）"

    # valuation
    if price is not None:
        pct_vs_mid, _ = valuation_metrics(price, fair_low, fair_high)
        val_label = "偏贵" if price >= fair_high else ("偏便宜" if price <= fair_low else "公允")
        val_str = f"{val_label} | 较中枢 {pct_vs_mid:+.1f}%" if pct_vs_mid is not None else "N/A"
    else:
        val_str = "N/A"

    cb_str = "N/A"
    if pack.cb:
        cb_3m, cb_prev3, cb_12m = pack.cb
        mom = cb_3m - cb_prev3
        cb_str = f"近3M {cb_3m:+.1f}t（较前3M {mom:+.1f}t）"

    # Compose (HTML)
    msg_lines: List[str] = []
    msg_lines.append(f"GoldTrendAlert | Liquidity {version}  {ts}")
    msg_lines.append(f"现价：{p_str}  | 14日波动带：{vol_str}")
    msg_lines.append("")
    msg_lines.append("<b>关键驱动（0–100）</b>")
    msg_lines.append(f"1 实际利率(10Y TIPS)：{s1}（WoW {wow}）→ {int(round(pack.s1))}")
    msg_lines.append(f"2 美元指数(广义)：30d {usd} → {int(round(pack.s2))}")
    msg_lines.append(f"3 BTC现货ETF净流(周)：{etf}（{pack.etf_src}）→ {int(round(pack.s3))}")
    msg_lines.append(f"4 10Y拍卖(BTC)：{auc_str} → {int(round(pack.s4))}")
    msg_lines.append(f"5 估值(公允 {int(fair_low)}–{int(fair_high)})：{val_str} → {int(round(pack.s5))}")
    msg_lines.append(f"6 央行购金(趋势)：{cb_str} → {int(round(pack.s6))}")
    msg_lines.append("")
    msg_lines.append("<b>行动（短版，可执行）</b>")
    plan_max = _to_float(env("PLAN_MAX_GOLD_WEIGHT", str(DEFAULT_PLAN_MAX))) or DEFAULT_PLAN_MAX
    msg_lines.extend(build_actions(pack, fair_low, fair_high, plan_max))
    msg_lines.append("")
    msg_lines.append(f"来源：现价=Stooq XAUUSD；ETF=CSV/SoSo；拍卖=US Treasury Fiscal Data；购金=WGC CSV")
    return "\n".join(msg_lines).strip()


def main() -> int:
    # Read config
    fred_key = env("FRED_API_KEY", DEFAULT_FRED_API_KEY)
    soso_key = env("SOSOVALUE_API_KEY", DEFAULT_SOSO_API_KEY)

    fair_low = _to_float(env("FAIR_BAND_LOW", str(DEFAULT_FAIR_LOW))) or DEFAULT_FAIR_LOW
    fair_high = _to_float(env("FAIR_BAND_HIGH", str(DEFAULT_FAIR_HIGH))) or DEFAULT_FAIR_HIGH

    btc_csv = env("BTC_ETF_FLOWS_CSV_URL", DEFAULT_BTC_ETF_CSV)
    wgc_csv = env("WGC_CSV_URL", DEFAULT_WGC_CSV)

    # price + vol
    price = stooq_xau_last_price()
    vol14, vol14_pct = compute_vol14_band(price)

    # Signal 1: DFII10 (10Y TIPS real yield)
    dfii_series = fred_series("DFII10", fred_key, limit=900)
    dfii_last = latest_non_null(dfii_series)
    dfii10 = dfii10_wow = None
    s1 = 50.0
    if dfii_last:
        d_last, v_last = dfii_last
        dfii10 = v_last
        back = nearest_on_or_before(dfii_series, d_last - timedelta(days=7))
        if back:
            dfii10_wow = v_last - back[1]
        # score: lower real yield => higher gold score
        # <=0.5 -> 85 ; >=2.5 -> 20 (linear)
        if v_last <= 0.5:
            s1 = 85.0
        elif v_last >= 2.5:
            s1 = 20.0
        else:
            s1 = score_linear(v_last, 0.5, 2.5, 85.0, 20.0)
        s1 = clamp(s1)

    # Signal 2: DTWEXBGS (broad dollar index) 30d change
    usd_series = fred_series("DTWEXBGS", fred_key, limit=1200)
    usd_last = latest_non_null(usd_series)
    usd_30d_pct = None
    s2 = 50.0
    if usd_last:
        d_last, v_last = usd_last
        back30 = nearest_on_or_before(usd_series, d_last - timedelta(days=30))
        if back30 and back30[1] not in (None, 0):
            usd_30d_pct = (v_last / back30[1] - 1.0) * 100.0
        if usd_30d_pct is not None:
            # +5% => 20; 0 => 50; -5% => 80
            if usd_30d_pct >= 0:
                s2 = score_linear(usd_30d_pct, 0.0, 5.0, 50.0, 20.0)
            else:
                s2 = score_linear(usd_30d_pct, -5.0, 0.0, 80.0, 50.0)
            s2 = clamp(s2)

    # Signal 3: BTC spot ETF weekly net flow (prefer CSV; fallback SoSo)
    etf_weekly_flow = btc_etf_weekly_flow_from_csv(btc_csv)
    etf_src = "CSV"
    if etf_weekly_flow is None:
        etf_weekly_flow = sosovalue_btc_etf_weekly_flow(soso_key)
        etf_src = "SoSo" if etf_weekly_flow is not None else "N/A"

    s3 = 50.0
    if etf_weekly_flow is not None:
        # inflow risk-on => lower gold score; outflow => higher gold score
        if etf_weekly_flow >= 2e9:
            s3 = 20.0
        elif etf_weekly_flow >= 0:
            s3 = score_linear(etf_weekly_flow, 0.0, 2e9, 55.0, 20.0)
        else:
            s3 = score_linear(etf_weekly_flow, -2e9, 0.0, 80.0, 55.0)
        s3 = clamp(s3)

    # Signal 4: 10Y auction BTC
    auc = treasury_10y_bid_to_cover()
    s4 = 50.0
    if auc:
        latest_btc, avg75, avg2y = auc
        delta = avg75 - avg2y
        # 弱于2y均值（delta<0）=> 需求变弱，利率压力可能更大 => gold score 下调
        # delta +0.10 => 60; delta -0.10 => 40
        s4 = clamp(score_linear(delta, -0.10, 0.10, 40.0, 60.0))

    # Signal 5: valuation
    pct_vs_mid, s5 = valuation_metrics(price, fair_low, fair_high)

    # Signal 6: central bank buy trend
    cb = wgc_netbuy_metrics(wgc_csv)
    s6 = 50.0
    if cb:
        cb_3m, cb_prev3, _ = cb
        mom = cb_3m - cb_prev3
        # mom +50t => 70; mom -50t => 40
        s6 = clamp(score_linear(mom, -50.0, 50.0, 40.0, 70.0))

    pack = Pack(
        price=price,
        vol14=vol14,
        vol14_pct=vol14_pct,
        s1=float(s1),
        s2=float(s2),
        s3=float(s3),
        s4=float(s4),
        s5=float(s5),
        s6=float(s6),
        dfii10=dfii10,
        dfii10_wow=dfii10_wow,
        usd_30d_pct=usd_30d_pct,
        etf_weekly_flow=etf_weekly_flow,
        etf_src=etf_src,
        auc=auc,
        cb=cb,
    )

    msg = build_message(pack, fair_low, fair_high)
    print(msg)

    ok = telegram_send_html(msg)
    if ok:
        print("Telegram sent OK.")
        return 0
    print("Telegram send failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
