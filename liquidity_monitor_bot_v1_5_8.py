# -*- coding: utf-8 -*-
"""GoldTrendAlert2 - Liquidity Monitor Bot v1.5.8

改动摘要（相对 v1.5.7）：
- 信号 1/2（DFII10, DTWEXBGS）不再依赖 FRED API KEY：改用 fredgraph.csv（无需 key）
- 信号 4（10Y 拍卖 Bid-to-Cover）尝试从 U.S. Treasury Fiscal Data API 自动拉取（失败则回退为中性 50 分）
- 信号 5（估值偏离）改为“公允区间”法（默认 3600–4200），确保不再 N/A
- 输出改为“短消息 + 强动作”：加入 ATR 止损线、加仓区、止盈位；删除“数据小表/诊断”

必填 Secrets：
- TELEGRAM_BOT_TOKEN
- TELEGRAM_CHAT_ID

可选参数（Environment variables）：
- FAIR_BAND_LOW：估值公允下沿（默认 3600）
- FAIR_BAND_HIGH：估值公允上沿（默认 4200）
- PLAN_MAX_GOLD_WEIGHT：计划黄金最大仓位（默认 18.0，单位：%）
- TAKE_PROFIT_LEVELS：止盈位（默认 "4600,4850,5050"）
- BUY_TRIGGER：回调到该价位以下才考虑加仓（默认 4100）
- BUY_BANDS：加仓分批区间（默认 "3960-3920,3920-3850,3850-3780"）

ETF 流向数据：优先 SoSoValue API（如配置），否则从 CSV 拉取。
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import requests

VERSION = "v1.5.8"
TIMEOUT = 25
RETRIES = 2
UA = {"User-Agent": f"GoldTrendAlert/{VERSION}"}


def env(key: str, default: Optional[str] = None) -> Optional[str]:
    v = os.environ.get(key)
    if v is None:
        return default
    v = str(v).strip()
    return v if v else default


def env_float(key: str, default: float) -> float:
    v = env(key)
    if v is None:
        return default
    try:
        return float(v)
    except Exception:
        return default


def now_cst_str() -> str:
    # 你之前的消息格式使用 CST(UTC+8)
    return datetime.now(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M CST")


def clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def fnum(x: Optional[float], digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "N/A"
    try:
        return f"{float(x):,.{digits}f}"
    except Exception:
        return str(x)


def fetch_text(url: str, params: Optional[dict] = None) -> Optional[str]:
    for i in range(RETRIES + 1):
        try:
            r = requests.get(url, params=params, headers=UA, timeout=TIMEOUT)
            r.raise_for_status()
            return r.text
        except Exception:
            if i >= RETRIES:
                return None
            time.sleep(1.2 * (i + 1))
    return None


def fetch_json(url: str, params: Optional[dict] = None, method: str = "GET", headers: Optional[dict] = None, body: Optional[str] = None) -> Optional[dict]:
    h = dict(UA)
    if headers:
        h.update(headers)
    for i in range(RETRIES + 1):
        try:
            if method.upper() == "POST":
                r = requests.post(url, params=params, headers=h, data=body, timeout=TIMEOUT)
            else:
                r = requests.get(url, params=params, headers=h, timeout=TIMEOUT)
            r.raise_for_status()
            return r.json()
        except Exception:
            if i >= RETRIES:
                return None
            time.sleep(1.2 * (i + 1))
    return None


# -------------------------
# Data sources
# -------------------------

def fred_series_csv(series_id: str, days: int = 365 * 5) -> List[Tuple[datetime, float]]:
    """Fetch FRED series via fredgraph.csv (无需 API KEY)。

    返回按日期升序排列的 [(dt, value), ...]
    """
    # fredgraph.csv 可直接带 id 参数
    # https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFII10
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    text = fetch_text(url, params={"id": series_id})
    if not text:
        return []
    out: List[Tuple[datetime, float]] = []
    try:
        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            d = row.get("DATE")
            v = row.get(series_id)
            if not d or not v or v.strip() in {".", ""}:
                continue
            try:
                dt = datetime.strptime(d.strip(), "%Y-%m-%d")
                val = float(v)
                out.append((dt, val))
            except Exception:
                continue
    except Exception:
        return []

    if not out:
        return []

    # 只取最近 days 天
    cutoff = datetime.utcnow() - timedelta(days=days)
    out = [(dt, val) for dt, val in out if dt >= cutoff]
    out.sort(key=lambda x: x[0])
    return out


@dataclass
class OHLC:
    date: datetime
    open: float
    high: float
    low: float
    close: float


def stooq_ohlc(symbol: str, max_rows: int = 4000) -> List[OHLC]:
    """Stooq CSV: Date,Open,High,Low,Close,Volume (日频)

    symbol 示例：xauusd
    """
    url = f"https://stooq.com/q/d/l/?s={symbol}&i=d"
    text = fetch_text(url)
    if not text:
        return []
    out: List[OHLC] = []
    try:
        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            try:
                dt = datetime.strptime(row["Date"], "%Y-%m-%d")
                o = float(row["Open"])
                h = float(row["High"])
                l = float(row["Low"])
                c = float(row["Close"])
                out.append(OHLC(dt, o, h, l, c))
            except Exception:
                continue
    except Exception:
        return []

    if not out:
        return []

    out.sort(key=lambda x: x.date)
    if len(out) > max_rows:
        out = out[-max_rows:]
    return out


def atr14(ohlc: List[OHLC], n: int = 14) -> Optional[float]:
    if len(ohlc) < n + 2:
        return None
    trs: List[float] = []
    for i in range(1, len(ohlc)):
        cur = ohlc[i]
        prev = ohlc[i - 1]
        tr = max(
            cur.high - cur.low,
            abs(cur.high - prev.close),
            abs(cur.low - prev.close),
        )
        trs.append(tr)
    if len(trs) < n:
        return None
    # 简单移动平均 ATR
    window = trs[-n:]
    return sum(window) / float(n)


# -------------------------
# ETF flow
# -------------------------

def _to_float(x) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().replace(",", "")
    if not s or s == ".":
        return None
    # (123) -> -123
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    try:
        return float(s)
    except Exception:
        return None


def ema(values: List[float], span: int) -> Optional[float]:
    if not values:
        return None
    if span <= 1:
        return values[-1]
    alpha = 2.0 / (span + 1.0)
    e = values[0]
    for v in values[1:]:
        e = alpha * v + (1.0 - alpha) * e
    return e


def linreg_slope(values: List[float]) -> float:
    """y=values, x=0..n-1 -> slope"""
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n
    num = 0.0
    den = 0.0
    for i, y in enumerate(values):
        dx = i - x_mean
        dy = y - y_mean
        num += dx * dy
        den += dx * dx
    return (num / den) if den != 0 else 0.0


def load_etf_flows() -> Tuple[List[Tuple[datetime, float]], str]:
    """返回 [(date, netflowUSD)...] 升序。"""

    # 1) SoSoValue API（可选）
    api_url = env("BTC_ETF_API_URL", "https://api.sosovalue.xyz/openapi/v2/etf/historicalInflowChart")
    api_method = (env("BTC_ETF_API_METHOD", "POST") or "POST").upper()

    # headers
    headers: Dict[str, str] = {}
    headers_json = env("BTC_ETF_API_HEADERS", "")
    if headers_json:
        try:
            headers.update(json.loads(headers_json))
        except Exception:
            pass
    if not any(k.lower() == "content-type" for k in headers):
        headers["Content-Type"] = "application/json"

    # api key
    if not any(k.lower() == "x-soso-api-key" for k in headers):
        k = env("SOSOVALUE_API_KEY")
        if k:
            headers["x-soso-api-key"] = k

    body = env("BTC_ETF_API_BODY", '{"type":"us-btc-spot"}')

    api_data = fetch_json(api_url, method=api_method, headers=headers, body=body)
    if api_data:
        rows = (api_data.get("data") or {}).get("list") or []
        if rows:
            # 尝试找日期字段与净流字段
            sample = rows[0]
            keys = {k.lower(): k for k in sample.keys()}
            dkey = None
            for k in ["date", "time", "day"]:
                if k in keys:
                    dkey = keys[k]
                    break
            vkey = None
            for k in ["totalnetinflow", "netinflow", "net_flow", "netflow", "total_net_inflow"]:
                if k in keys:
                    vkey = keys[k]
                    break

            out: List[Tuple[datetime, float]] = []
            if dkey and vkey:
                for r in rows:
                    dv = r.get(dkey)
                    vv = _to_float(r.get(vkey))
                    if vv is None or dv is None:
                        continue
                    try:
                        dt = datetime.fromisoformat(str(dv).replace("Z", "+00:00")).replace(tzinfo=None)
                    except Exception:
                        try:
                            dt = datetime.strptime(str(dv)[:10], "%Y-%m-%d")
                        except Exception:
                            continue
                    out.append((dt, float(vv)))

            if out:
                out.sort(key=lambda x: x[0])
                return out, "API"

    # 2) CSV fallback
    csv_url = env("BTC_ETF_FLOWS_CSV_URL") or (
        "https://raw.githubusercontent.com/InferFlux/GoldTrendAlert2/refs/heads/main/.bot_state/btc_spot_etf_flows.csv"
    )
    text = fetch_text(csv_url)
    if not text:
        return [], "CSV(N/A)"

    date_field = env("BTC_ETF_CSV_DATE_FIELD", "Date")
    flow_field = env("BTC_ETF_CSV_FLOW_FIELD", "NetFlowUSD")

    out2: List[Tuple[datetime, float]] = []
    try:
        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            if date_field not in row or flow_field not in row:
                # 尝试自动匹配
                lower = {k.lower(): k for k in row.keys()}
                date_field = lower.get("date", date_field)
                flow_field = lower.get("netflowusd", flow_field)
            dv = row.get(date_field)
            fv = _to_float(row.get(flow_field))
            if dv is None or fv is None:
                continue
            try:
                dt = datetime.strptime(str(dv)[:10], "%Y-%m-%d")
            except Exception:
                continue
            out2.append((dt, float(fv)))
    except Exception:
        return [], "CSV(N/A)"

    out2.sort(key=lambda x: x[0])
    return out2, "CSV"


def etf_flow_score(flows: List[Tuple[datetime, float]]) -> Tuple[float, Optional[float], str]:
    if len(flows) < 10:
        return 50.0, None, "ETF: 数据不足"

    values = [v for _, v in flows]
    n = len(values)

    long_span = int(env_float("ETF_LONG_D", 34))
    short_span = int(env_float("ETF_SHORT_D", 13))

    # 自适应：样本太短则缩窗
    if n < max(long_span, short_span):
        for l, s in [(21, 8), (13, 5)]:
            if n >= max(l, s):
                long_span, short_span = l, s
                break

    ema_l = ema(values, long_span)
    ema_s = ema(values, short_span)
    trend_ok = (ema_s is not None and ema_l is not None and ema_s > ema_l)

    win = min(10, n)
    slope = linreg_slope(values[-win:])  # USD/day

    score = 50.0 + (10.0 if trend_ok else -10.0) + clip(slope / 1e6, -10.0, 10.0)
    score = clip(score, 0.0, 100.0)

    # 近 7 天净流（按日期过滤）
    end = flows[-1][0]
    start = end - timedelta(days=7)
    weekly = sum(v for dt, v in flows if dt >= start)

    diag = f"ETF: span={long_span}/{short_span}, slope={slope:.2f}"  # 不会发到 Telegram，仅内部用
    return float(score), float(weekly), diag


# -------------------------
# Auction Bid-to-Cover (10Y Note)
# -------------------------

def fetch_10y_bid_to_cover() -> Optional[Dict[str, float]]:
    """从 Treasury Fiscal Data API 拉取 10Y Note 拍卖 Bid-to-Cover。

    若成功，返回：
    {
      "btc_last": ...,
      "btc_recent_avg": ...,
      "btc_long_avg": ...,
      "delta": ...,
    }
    """
    base = "https://api.fiscaldata.treasury.gov/services/api/fiscal_service/v1/accounting/od/auctions_query"
    start_long = (datetime.utcnow() - timedelta(days=730)).strftime("%Y-%m-%d")

    # Fiscal Data API 过滤语法：field:operator:value，多个条件逗号分隔
    # 这里用 Note + 10-Year
    filter_ = f"security_type:eq:Note,security_term:eq:10-Year,auction_date:gte:{start_long}"
    params = {
        "filter": filter_,
        "sort": "-auction_date",
        "page[size]": "100",
    }

    j = fetch_json(base, params=params)
    if not j:
        return None

    data = j.get("data") or []
    if not data:
        return None

    # 动态识别字段名
    sample = data[0]
    key_map = {k.lower(): k for k in sample.keys()}
    date_key = key_map.get("auction_date") or key_map.get("auctiondate") or "auction_date"

    btc_key = None
    for lk, ok in key_map.items():
        if "bid" in lk and "cover" in lk:
            btc_key = ok
            break
    if not btc_key:
        # 某些表可能用 bid_to_cover_ratio / bid_to_cover
        for cand in ["bid_to_cover_ratio", "bid_to_cover", "bid_to_cover_r"]:
            if cand in key_map:
                btc_key = key_map[cand]
                break
    if not btc_key:
        return None

    rows: List[Tuple[datetime, float]] = []
    for r in data:
        dv = r.get(date_key)
        bv = _to_float(r.get(btc_key))
        if not dv or bv is None:
            continue
        try:
            dt = datetime.strptime(str(dv)[:10], "%Y-%m-%d")
        except Exception:
            continue
        rows.append((dt, float(bv)))

    if not rows:
        return None

    # 最新在前
    rows.sort(key=lambda x: x[0], reverse=True)

    btc_last = rows[0][1]

    cutoff_recent = datetime.utcnow() - timedelta(days=75)
    recent = [v for dt, v in rows if dt >= cutoff_recent]
    long = [v for _, v in rows[:24]]  # 约 2 年

    if not recent:
        recent = long[:3] if long else []

    if not long:
        return None

    btc_recent_avg = sum(recent) / len(recent)
    btc_long_avg = sum(long) / len(long)
    delta = btc_recent_avg - btc_long_avg

    return {
        "btc_last": float(btc_last),
        "btc_recent_avg": float(btc_recent_avg),
        "btc_long_avg": float(btc_long_avg),
        "delta": float(delta),
    }


# -------------------------
# Scoring + Message
# -------------------------

def score_real_yield(dfii10: Optional[float], wow: Optional[float]) -> float:
    # 实际利率越低越利多黄金；实际利率上行也偏空
    if dfii10 is None:
        return 50.0
    wow_v = wow or 0.0
    return float(clip(50.0 + (-dfii10) * 10.0 + (-wow_v) * 5.0, 0.0, 100.0))


def score_dxy_change(dxy_30d_pct: Optional[float]) -> float:
    # 美元走弱利多黄金
    if dxy_30d_pct is None:
        return 50.0
    return float(clip(50.0 + (-dxy_30d_pct) * 3.0, 0.0, 100.0))


def score_auction(delta_btc: Optional[float]) -> float:
    if delta_btc is None:
        return 50.0
    # delta 典型量级 ~ [-0.5, 0.5]，乘 50 得到 +/-25 分
    return float(clip(50.0 + delta_btc * 50.0, 0.0, 100.0))


def score_valuation(spot: Optional[float], low: float, high: float) -> Tuple[float, Optional[float]]:
    if spot is None or low <= 0 or high <= 0 or high <= low:
        return 50.0, None
    mid = (low + high) / 2.0
    half = (high - low) / 2.0
    z = (spot - mid) / half  # >0 表示偏贵
    val_points = clip(-z * 20.0, -20.0, 20.0)  # 偏贵给负分
    score = clip(50.0 + val_points * 2.5, 0.0, 100.0)
    dev_pct = (spot / mid - 1.0) * 100.0
    return float(score), float(dev_pct)


def format_arrow(score: float) -> str:
    # 简单符号：>55 认为偏多，<45 偏空
    if score >= 55:
        return "↑"
    if score <= 45:
        return "↓"
    return "→"


def main() -> int:
    # 价格与 ATR
    ohlc = stooq_ohlc("xauusd")
    spot = ohlc[-1].close if ohlc else None
    atr = atr14(ohlc, 14) if ohlc else None

    # FRED：DFII10、DTWEXBGS、DGS10、VIXCLS
    dfii10_series = fred_series_csv("DFII10", days=365 * 2)
    dtwex_series = fred_series_csv("DTWEXBGS", days=365 * 2)

    dgs10_series = fred_series_csv("DGS10", days=365 * 1)
    vix_series = fred_series_csv("VIXCLS", days=365 * 1)

    def last_val(series: List[Tuple[datetime, float]]) -> Optional[float]:
        return series[-1][1] if series else None

    def wow_val(series: List[Tuple[datetime, float]], k: int = 5) -> Optional[float]:
        if len(series) < k + 1:
            return None
        return series[-1][1] - series[-(k + 1)][1]

    dfii10 = last_val(dfii10_series)
    dfii10_wow = wow_val(dfii10_series, 5)

    dgs10 = last_val(dgs10_series)
    dgs10_wow = wow_val(dgs10_series, 5)

    vix = last_val(vix_series)
    vix_wow = wow_val(vix_series, 5)

    dxy_last = last_val(dtwex_series)
    dxy_30d = None
    if len(dtwex_series) >= 31:
        dxy_30d = (dtwex_series[-1][1] / dtwex_series[-31][1] - 1.0) * 100.0

    # ETF
    flows, etf_src = load_etf_flows()
    sc_etf, weekly_flow, _diag = etf_flow_score(flows) if flows else (50.0, None, "ETF: N/A")

    # Auction
    auc = fetch_10y_bid_to_cover()
    delta_btc = auc["delta"] if auc else None
    # Optional manual override (backward compatible with old workflow)
    if delta_btc is None:
        ov = env("UST_AUCTIONS_DELTA")
        if ov is not None:
            try:
                delta_btc = float(ov)
            except Exception:
                pass
    sc_auc = score_auction(delta_btc)

    # Valuation (fair band)
    fair_low = env_float("FAIR_BAND_LOW", 3600.0)
    fair_high = env_float("FAIR_BAND_HIGH", 4200.0)
    sc_val, dev_mid_pct = score_valuation(spot, fair_low, fair_high)

    # Scores
    sc_tips = score_real_yield(dfii10, dfii10_wow)
    sc_dxy = score_dxy_change(dxy_30d)

    # Composite
    scores = [sc_tips, sc_dxy, sc_etf, sc_auc, sc_val]
    comp = sum(scores) / len(scores)

    # Action / regime
    plan_max = env_float("PLAN_MAX_GOLD_WEIGHT", 18.0)
    tp_levels = env("TAKE_PROFIT_LEVELS", "4600,4850,5050")
    take_profits: List[float] = []
    for p in tp_levels.split(","):
        try:
            take_profits.append(float(p.strip()))
        except Exception:
            pass
    take_profits = sorted(set(take_profits))

    buy_trigger = env_float("BUY_TRIGGER", 4100.0)
    buy_bands_raw = env("BUY_BANDS", "3960-3920,3920-3850,3850-3780")

    # Stops
    stop_1 = stop_15 = stop_2 = None
    if spot is not None and atr is not None:
        stop_1 = spot - 1.0 * atr
        stop_15 = spot - 1.5 * atr
        stop_2 = spot - 2.0 * atr

    # 文字结论（更强动作）
    if comp >= 65:
        stance = "偏多"
        action_main = "趋势仍强：持有为主，回撤到加仓区再加。"
    elif comp >= 55:
        stance = "偏多/中性"
        action_main = "以持有为主：不追高，回撤再加。"
    elif comp >= 45:
        stance = "中性"
        action_main = "观望偏多：仓位不动，盯好止损线/回撤加仓区。"
    else:
        stance = "谨慎"
        action_main = "偏谨慎：优先风控与锁利润，避免追高加仓。"

    # 估值补充
    val_hint = ""
    if spot is not None:
        if spot > fair_high:
            val_hint = "估值偏贵（高于公允上沿）：更应‘不追高/先锁利润’。"
        elif spot < fair_low:
            val_hint = "估值偏便宜（低于公允下沿）：更适合‘分批加’。"
        else:
            val_hint = "估值处于公允区间：按趋势与风控执行。"

    # ETF hint
    etf_hint = ""
    if weekly_flow is not None:
        if weekly_flow <= -1e9:
            etf_hint = "资金面偏空（ETF 周净流出显著）。"
        elif weekly_flow >= 1e9:
            etf_hint = "资金面偏多（ETF 周净流入显著）。"
        else:
            etf_hint = "资金面中性（ETF 周流向不极端）。"

    # Take profit hint
    tp_hint = ""
    if spot is not None and take_profits:
        above = [p for p in take_profits if spot >= p]
        if above:
            tp_hint = f"已触及/突破止盈位 {', '.join(str(int(p)) for p in above)}：建议分批落袋。"
        else:
            nxt = next((p for p in take_profits if spot < p), None)
            if nxt:
                tp_hint = f"上方首个止盈位：{int(nxt)}（到位分批落袋）。"

    # Build message (short)
    lines: List[str] = []
    lines.append(f"GoldTrendAlert | Liquidity {VERSION}  {now_cst_str()}")
    if spot is not None:
        atr_pct = (atr / spot * 100.0) if (atr is not None and spot != 0) else None
        if atr is not None:
            lines.append(f"现价：${fnum(spot,2)}  | ATR14：{fnum(atr,1)}（{fnum(atr_pct,2)}%）")
        else:
            lines.append(f"现价：${fnum(spot,2)}")
    lines.append(f"综合：{int(round(comp))}/100（{stance}）")

    # Signals (compact)
    lines.append("\n关键驱动（0–100）：")
    lines.append(f"1 实际利率 DFII10：{fnum(dfii10,2)}（WoW {fnum(dfii10_wow,2)}） {format_arrow(sc_tips)} {fnum(sc_tips,0)}")
    lines.append(f"2 广义美元 DTWEXBGS：30d {fnum(dxy_30d,2)}% {format_arrow(sc_dxy)} {fnum(sc_dxy,0)}")
    if weekly_flow is not None:
        lines.append(f"3 ETF(周净流)：{fnum(weekly_flow,0)}（{etf_src}） {format_arrow(sc_etf)} {fnum(sc_etf,0)}")
    else:
        lines.append(f"3 ETF：N/A（{etf_src}） → 50")
    if auc is not None:
        lines.append(
            f"4 10Y拍卖BTC：{fnum(auc['btc_last'],2)}（近75d均 {fnum(auc['btc_recent_avg'],2)} vs 2y均 {fnum(auc['btc_long_avg'],2)}，Δ {fnum(auc['delta'],2)}） {format_arrow(sc_auc)} {fnum(sc_auc,0)}"
        )
    else:
        lines.append(f"4 10Y拍卖BTC：暂不可用 → 50")
    lines.append(
        f"5 估值（公允 {int(fair_low)}–{int(fair_high)}）：相对中枢偏离 {fnum(dev_mid_pct,1)}% {format_arrow(sc_val)} {fnum(sc_val,0)}"
    )

    # Action (short + meaningful)
    lines.append("\n行动（短版，中文执行）：")
    lines.append(f"- {action_main}")
    if val_hint:
        lines.append(f"- {val_hint}")
    if etf_hint:
        lines.append(f"- {etf_hint}")
    if tp_hint:
        lines.append(f"- {tp_hint}")

    if stop_15 is not None:
        lines.append(
            f"- 风控止损（默认 1.5xATR）：{fnum(stop_15,0)}  | 备选：1.0x {fnum(stop_1,0)} / 2.0x {fnum(stop_2,0)}"
        )
        lines.append("  规则：若‘收盘价’跌破你的止损线 → 按计划减仓 50–100%。")

    # Buy guidance
    if spot is not None:
        if spot <= buy_trigger:
            lines.append(f"- 加仓：已在触发线下（<= {int(buy_trigger)}），按区间分批：{buy_bands_raw}")
        else:
            lines.append(f"- 加仓：只在回调到 <= {int(buy_trigger)} 再考虑（分批区间：{buy_bands_raw}）。")

    lines.append(f"- 仓位纪律：计划黄金上限 {plan_max:.1f}%（若已超上限，优先减到 ≤ 上限）。")

    text = "\n".join(lines)

    bot = env("TELEGRAM_BOT_TOKEN")
    chat = env("TELEGRAM_CHAT_ID")

    if bot and chat:
        try:
            resp = requests.post(
                f"https://api.telegram.org/bot{bot}/sendMessage",
                json={"chat_id": chat, "text": text},
                timeout=TIMEOUT,
            )
            # 打印可诊断信息（不泄露 token）
            if resp.status_code >= 400:
                print("Telegram status:", resp.status_code, file=sys.stderr)
                print("Telegram resp:", resp.text[:500], file=sys.stderr)
                resp.raise_for_status()
        except Exception as e:
            print("发送Telegram失败：", repr(e), file=sys.stderr)
            print(text)
    else:
        # 本地/调试：无 token 时直接输出
        print(text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
