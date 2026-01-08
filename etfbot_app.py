"""ETFBot 的 Panel 应用模块。

该文件从原 notebook 抽离，便于维护与复用。
"""

from __future__ import annotations

import asyncio
import datetime
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import akshare as ak
import pandas as pd
import panel as pn

import llm_client


# 默认参数：不使用环境变量（UI 中可随时调整）
DEFAULT_LOOKBACK_DAYS = 120
DEFAULT_RECENT_ROWS = 10
DEFAULT_NEWS_ROWS = 5

# 轻量裁剪上下文：只裁剪发给 LLM 的 messages，不影响界面历史展示
MAX_CONTEXT_MESSAGES = 20

SYSTEM_PROMPT = """
你是 ETFBot，一位【ETF 场内基金优先】的日线复盘与交易计划助手，面向【短线波段】用户。
也支持输入 6 位 A 股股票代码做同样的技术面复盘与计划，但请注意：输出风格与关键位/情绪判断更偏 ETF/指数类（个股需自行结合公告、基本面与事件风险）。
默认工作流：收盘后复盘，开盘前给出次日计划/预测（以“条件单思维”描述，不做绝对保证）。

你会收到：
- 最近 N 行日线行情（原始数据）
- 指标摘要（MA/RSI/MACD/波动/量能/关键位等）
-（可选）相关新闻摘要

输出要求（务必结构化，简洁可执行）：
1) 【复盘结论】基于当日收盘数据，1-2 句概括趋势与波动（基于指标摘要，不要泛泛而谈）。
2) 【关键位】给出支撑/压力（优先用 20/60 日高低、MA20/MA60），并说明“突破/跌破”的意义。
3) 【量能与情绪】结合量比/成交量均值，判断是否放量/缩量；若有新闻，判断是否可能影响短线情绪。
4) 【开盘前计划/预测（次日）】给出 1-3 条动作建议，每条包含：
   - 触发条件（满足什么才做；如高开/低开/突破/回踩/量能配合）
   - 风控/失效条件（跌破哪里算走坏）
   - 仓位建议（轻/中/重或分批）
5) 【风险提示】提醒数据源与市场风险，不给出绝对保证。
""".strip()


def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """从候选列名中，返回第一个在 DataFrame 里存在的列名。"""
    for name in candidates:
        if name in df.columns:
            return name
    return None


def _coerce_numeric(series: pd.Series) -> pd.Series:
    """将序列尽可能转换为数值类型（不可转换则为 NaN）。"""
    return pd.to_numeric(series, errors="coerce")


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """将 AkShare 返回的日线数据统一为 date/open/high/low/close/volume，并按日期升序。"""
    if df is None or df.empty:
        return pd.DataFrame()

    date_col = _pick_col(df, ["日期", "交易日期", "时间", "date"])
    open_col = _pick_col(df, ["开盘", "开盘价", "open"])
    close_col = _pick_col(df, ["收盘", "收盘价", "最新价", "close"])
    high_col = _pick_col(df, ["最高", "最高价", "high"])
    low_col = _pick_col(df, ["最低", "最低价", "low"])
    vol_col = _pick_col(df, ["成交量", "成交量(手)", "成交量(股)", "volume"])

    out = pd.DataFrame()
    if date_col:
        out["date"] = pd.to_datetime(df[date_col], errors="coerce")
    if open_col:
        out["open"] = _coerce_numeric(df[open_col])
    if high_col:
        out["high"] = _coerce_numeric(df[high_col])
    if low_col:
        out["low"] = _coerce_numeric(df[low_col])
    if close_col:
        out["close"] = _coerce_numeric(df[close_col])
    if vol_col:
        out["volume"] = _coerce_numeric(df[vol_col])

    if "date" in out.columns:
        out = out.dropna(subset=["date"]).sort_values("date")
    else:
        out = out.reset_index(drop=True)

    return out.reset_index(drop=True)


def _format_recent_quotes(df: pd.DataFrame, n: int = 5) -> str:
    """将最近 N 行行情整理成 Markdown 表格，供界面展示/传入提示词。"""
    if df is None or df.empty:
        return "（未获取到行情数据）"

    date_col = _pick_col(df, ["日期", "交易日期", "时间", "date"])
    open_col = _pick_col(df, ["开盘", "开盘价", "open"])
    close_col = _pick_col(df, ["收盘", "收盘价", "最新价", "close"])
    high_col = _pick_col(df, ["最高", "最高价", "high"])
    low_col = _pick_col(df, ["最低", "最低价", "low"])
    vol_col = _pick_col(df, ["成交量", "成交量(手)", "成交量(股)", "volume"])

    use_cols = [
        c for c in [date_col, open_col, close_col, high_col, low_col, vol_col] if c
    ]
    view = df.copy()
    if date_col:
        view = view.sort_values(by=date_col, ascending=False)
    view = view.head(n)
    if use_cols:
        view = view[use_cols]
    return view.to_markdown(index=False)


def _fetch_quotes(symbol_6digit: str, lookback_days: int):
    """拉取标的日线行情（近 lookback_days 天）。

    策略：
    - 优先尝试 `ETF_zh_a_hist`（部分 AkShare 版本更宽松）
    - 失败则回退 `fund_etf_hist_em`（更偏 ETF）
    """

    start_date = (
        datetime.datetime.now() - datetime.timedelta(days=lookback_days)
    ).strftime("%Y%m%d")
    end_date = datetime.datetime.now().strftime("%Y%m%d")

    try:
        df = ak.ETF_zh_a_hist(
            symbol=symbol_6digit,
            period="daily",
            start_date=start_date,
            adjust="qfq",
        )
        if df is not None and not df.empty:
            return df
    except Exception:
        pass

    try:
        df = ak.fund_etf_hist_em(
            symbol=symbol_6digit,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust="qfq",
        )
        if df is not None and not df.empty:
            return df
    except Exception:
        pass

    return None


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """计算 RSI 指标（默认 14）。"""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """计算 MACD（DIF/DEA/HIST）。"""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    dif = ema_fast - ema_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = dif - dea
    return dif, dea, hist


def _atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """计算 ATR 指标（默认 14）。"""
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def _compute_indicator_summary(quotes_df: pd.DataFrame, lookback_days: int) -> str:
    """基于行情计算指标摘要（MA/RSI/MACD/ATR/关键位/量能等），返回 Markdown 文本。"""
    norm = _normalize_ohlcv(quotes_df)
    if norm.empty or "close" not in norm.columns:
        return "（指标：数据不足，无法计算）"

    close = norm["close"]
    high = norm["high"] if "high" in norm.columns else close
    low = norm["low"] if "low" in norm.columns else close
    vol = norm["volume"] if "volume" in norm.columns else pd.Series([pd.NA] * len(norm))

    latest_close = float(close.iloc[-1]) if pd.notna(close.iloc[-1]) else None

    ma5 = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()

    rsi14 = _rsi(close, 14)
    dif, dea, hist = _macd(close, 12, 26, 9)
    atr14 = _atr(high, low, close, 14)

    vol_ma5 = vol.rolling(5).mean() if vol.notna().any() else None
    vol_ma20 = vol.rolling(20).mean() if vol.notna().any() else None

    hi20 = high.rolling(20).max()
    lo20 = low.rolling(20).min()
    hi60 = high.rolling(60).max()
    lo60 = low.rolling(60).min()

    def _fmt(x: Any, digits: int = 4) -> str:
        if x is None or pd.isna(x):
            return "NA"
        try:
            return f"{float(x):.{digits}f}"
        except Exception:
            return "NA"

    def _fmt_int(x: Any) -> str:
        if x is None or pd.isna(x):
            return "NA"
        try:
            return f"{int(float(x)):,}"
        except Exception:
            return "NA"

    latest: Dict[str, Any] = {
        "MA5": ma5.iloc[-1] if len(ma5) else pd.NA,
        "MA10": ma10.iloc[-1] if len(ma10) else pd.NA,
        "MA20": ma20.iloc[-1] if len(ma20) else pd.NA,
        "MA60": ma60.iloc[-1] if len(ma60) else pd.NA,
        "RSI14": rsi14.iloc[-1] if len(rsi14) else pd.NA,
        "DIF": dif.iloc[-1] if len(dif) else pd.NA,
        "DEA": dea.iloc[-1] if len(dea) else pd.NA,
        "MACD_HIST": hist.iloc[-1] if len(hist) else pd.NA,
        "ATR14": atr14.iloc[-1] if len(atr14) else pd.NA,
        "VOL": vol.iloc[-1] if len(vol) else pd.NA,
        "VOL_MA5": vol_ma5.iloc[-1] if vol_ma5 is not None and len(vol_ma5) else pd.NA,
        "VOL_MA20": (
            vol_ma20.iloc[-1] if vol_ma20 is not None and len(vol_ma20) else pd.NA
        ),
        "HI20": hi20.iloc[-1] if len(hi20) else pd.NA,
        "LO20": lo20.iloc[-1] if len(lo20) else pd.NA,
        "HI60": hi60.iloc[-1] if len(hi60) else pd.NA,
        "LO60": lo60.iloc[-1] if len(lo60) else pd.NA,
    }

    vol_ratio = (
        (latest["VOL"] / latest["VOL_MA20"])
        if pd.notna(latest["VOL"])
        and pd.notna(latest["VOL_MA20"])
        and float(latest["VOL_MA20"]) != 0
        else pd.NA
    )

    trend_hint = ""
    if latest_close is not None:
        above20 = pd.notna(latest["MA20"]) and latest_close >= float(latest["MA20"])
        above60 = pd.notna(latest["MA60"]) and latest_close >= float(latest["MA60"])
        if above20 and above60:
            trend_hint = "偏强（站上 MA20/MA60）"
        elif above20 and not above60:
            trend_hint = "中性偏强（站上 MA20，仍在 MA60 附近）"
        elif (not above20) and above60:
            trend_hint = "中性偏弱（跌破 MA20，仍在 MA60 上方）"
        else:
            trend_hint = "偏弱（位于 MA20/MA60 下方）"

    def _pct_to(level: Any) -> str:
        if latest_close is None or pd.isna(level) or float(level) == 0:
            return "NA"
        return f"{(latest_close / float(level) - 1) * 100:.2f}%"

    lines: List[str] = []
    lines.append(
        f"- 回看：{lookback_days}天（日线：收盘后复盘；开盘前给次日计划/预测）"
    )
    lines.append(f"- 最新收盘：{_fmt(latest_close, 4)}；趋势：{trend_hint or 'NA'}")
    lines.append(
        "- 均线："
        f"MA5={_fmt(latest['MA5'])} / MA10={_fmt(latest['MA10'])} / MA20={_fmt(latest['MA20'])} / MA60={_fmt(latest['MA60'])}"
    )
    lines.append(f"- RSI14：{_fmt(latest['RSI14'], 2)}")
    lines.append(
        "- MACD："
        f"DIF={_fmt(latest['DIF'], 4)} DEA={_fmt(latest['DEA'], 4)} HIST={_fmt(latest['MACD_HIST'], 4)}"
    )
    lines.append(f"- 波动(ATR14)：{_fmt(latest['ATR14'], 4)}")

    if pd.notna(vol_ratio):
        lines.append(
            "- 量能："
            f"VOL={_fmt_int(latest['VOL'])}  VOL_MA5={_fmt_int(latest['VOL_MA5'])}  VOL_MA20={_fmt_int(latest['VOL_MA20'])}  量比≈{_fmt(vol_ratio, 2)}"
        )

    lines.append(
        "- 关键位："
        f"20日高={_fmt(latest['HI20'])}（距今{_pct_to(latest['HI20'])}） / 20日低={_fmt(latest['LO20'])}（距今{_pct_to(latest['LO20'])}）"
    )
    lines.append(
        "- 关键位："
        f"60日高={_fmt(latest['HI60'])}（距今{_pct_to(latest['HI60'])}） / 60日低={_fmt(latest['LO60'])}（距今{_pct_to(latest['LO60'])}）"
    )

    return "\n".join(lines)


def _fetch_related_news(symbol_6digit: str) -> Optional[pd.DataFrame]:
    """拉取该代码相关新闻（尽力而为：接口可能为空或变更）。"""
    try:
        df = ak.stock_news_em(symbol=symbol_6digit)
        if df is not None and not df.empty:
            return df
    except Exception:
        pass
    return None


def _format_recent_news(df: Optional[pd.DataFrame], n: int = 5) -> str:
    """将新闻 DataFrame 整理为 Markdown 列表（最多 n 条）。"""
    if df is None or df.empty:
        return "（未获取到相关新闻）"

    time_col = _pick_col(df, ["发布时间", "时间", "日期", "publish_time", "pub_time"])
    title_col = _pick_col(df, ["标题", "新闻标题", "title"])
    source_col = _pick_col(df, ["来源", "文章来源", "source"])
    url_col = _pick_col(df, ["链接", "url", "新闻链接"])

    view = df.copy().head(n)
    lines: List[str] = []
    for _, row in view.iterrows():
        parts: List[str] = []
        if time_col and str(row.get(time_col, "")).strip():
            parts.append(str(row.get(time_col)).strip())
        if source_col and str(row.get(source_col, "")).strip():
            parts.append(str(row.get(source_col)).strip())
        prefix = " · ".join(parts)
        title = str(row.get(title_col, "")).strip() if title_col else ""
        url = str(row.get(url_col, "")).strip() if url_col else ""

        if title and url:
            item = f"- {prefix}：[{title}]({url})" if prefix else f"- [{title}]({url})"
        elif title:
            item = f"- {prefix}：{title}" if prefix else f"- {title}"
        else:
            continue
        lines.append(item)

    return "\n".join(lines) if lines else "（未获取到相关新闻）"


@dataclass
class _ComputeResult:
    """一次“输入代码并计算”的结构化结果，用于 UI 展示与写入上下文。"""

    latest_md: str
    indicator_summary: str
    news_md: str
    prompt: Optional[str]
    ai_response: str
    news_note: str


class ETFBotApp:
    def __init__(self):
        """初始化应用状态与 Panel 组件，并组装 dashboard。"""
        pn.extension()

        self.client = llm_client.make_gemini_openai_client()

        self.current_symbol: Optional[str] = None
        self.panels: List[Any] = []
        self.context: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

        self.active_request_id = 0

        # 组件（输入框/按钮/参数）
        self.inp = pn.widgets.TextInput(
            value="",
            placeholder="请先单独输入 6 位 ETF/股票代码开始会话，然后再提问",
            width=400,
        )
        self.btn = pn.widgets.Button(name="发送", button_type="primary")
        self.reset_btn = pn.widgets.Button(name="重置会话", button_type="warning")

        self.lookback_days_input = pn.widgets.IntInput(
            name="回看天数",
            value=DEFAULT_LOOKBACK_DAYS,
            start=10,
            end=365,
            step=5,
            width=140,
        )
        self.recent_rows_input = pn.widgets.IntInput(
            name="展示行数",
            value=DEFAULT_RECENT_ROWS,
            start=3,
            end=30,
            step=1,
            width=140,
        )
        self.news_rows_input = pn.widgets.IntInput(
            name="新闻条数",
            value=DEFAULT_NEWS_ROWS,
            start=0,
            end=20,
            step=1,
            width=140,
        )

        self.status_pane = pn.pane.Markdown(self._render_status(), width=280)
        self.conversation_box = pn.Column()

        self.reset_btn.on_click(self.handle_reset)
        self.btn.on_click(self._schedule_send)

        self.dashboard = pn.Column(
            pn.pane.Markdown(
                "## 📊 ETFBot（收盘后复盘 · 开盘前计划/预测 · 简单易维护）"
            ),
            pn.pane.Markdown(
                "输入 6 位代码（ETF/A 股 均可）。注意：该工具以 ETF 场内基金为主要设计场景；个股也能跑技术面复盘，但请更关注公告/事件风险并更保守执行。\n\n"
                "注意：展示的行情/指标/新闻会作为提示词的一部分传入 ETFBot（不会自动传入全部回看天数数据）。\n\n"
                "使用方式：收盘后复盘，开盘前查看次日计划/预测（以触发条件 + 风控为主）。"
            ),
            pn.Row(self.inp, self.btn, self.reset_btn),
            pn.Row(
                self.lookback_days_input,
                self.recent_rows_input,
                self.news_rows_input,
                self.status_pane,
            ),
            self.conversation_box,
        )

        self._refresh_view()

    def reset_session(self, symbol: Optional[str] = None) -> None:
        """重置会话：清空上下文与对话面板，可选保留/设置当前标的。"""
        self.context = [{"role": "system", "content": SYSTEM_PROMPT}]
        self.panels = []
        self.current_symbol = symbol

    def trim_context(self, max_messages: int = MAX_CONTEXT_MESSAGES) -> None:
        """裁剪上下文历史，避免发给模型的 messages 过长（保留 system + 最近 N 条）。"""
        if not isinstance(self.context, list) or not self.context:
            self.context = [{"role": "system", "content": SYSTEM_PROMPT}]
            return
        if len(self.context) <= 1 + max_messages:
            return
        self.context = self.context[:1] + self.context[-max_messages:]

    def _trim_messages_for_llm(
        self, messages: List[Dict[str, str]], max_messages: int = MAX_CONTEXT_MESSAGES
    ):
        """仅对“本次要提交给模型”的 messages 做轻量裁剪，不影响 UI 展示历史。"""
        if not isinstance(messages, list) or not messages:
            return [{"role": "system", "content": SYSTEM_PROMPT}]
        if len(messages) <= 1 + max_messages:
            return messages
        return messages[:1] + messages[-max_messages:]

    def _render_status(self) -> str:
        """渲染右侧状态栏：是否检测到大模型 API Key。"""
        model_name = getattr(llm_client, "DEFAULT_MODEL", "")
        if self.client is not None:
            return f"**LLM(OpenAI-compatible)**：已配置 `{model_name}`"
        return (
            "**LLM(OpenAI-compatible)**：未配置 `LLM_API_KEY`（将无法调用模型）\n"
            f"**DEFAULT_MODEL**：`{model_name}`"
        )

    def get_completion_from_messages(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 1.0,
    ) -> str:
        """调用大模型生成回复（若未配置 key，则返回提示信息）。"""
        return llm_client.chat_completion(
            client=self.client,
            messages=messages,
            model=model,
            temperature=temperature,
        )

    def _refresh_view(self) -> None:
        """将内部 panels 同步到 conversation_box（用于刷新界面）。"""
        self.conversation_box.objects = list(self.panels)

    def _set_processing(self, is_processing: bool) -> None:
        """设置“处理中”状态：禁用输入与发送，并切换按钮文案。"""
        if is_processing:
            self.btn.name = "处理中…"
            self.btn.disabled = True
            self.inp.disabled = True
        else:
            self.btn.name = "发送"
            self.btn.disabled = False
            self.inp.disabled = False

    def handle_reset(self, _) -> None:
        """处理“重置会话”：递增 request_id，防止旧任务回写界面。"""
        self.active_request_id += 1
        self.reset_session(None)
        self._set_processing(False)
        self._refresh_view()

    async def _run_in_thread(self, func, *args):
        """在线程池中执行阻塞/耗时函数，避免卡死 Panel UI 事件循环。"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, func, *args)

    def _compute_for_code(
        self,
        symbol_6digit: str,
        lookback_days: int,
        recent_rows: int,
        news_rows: int,
        base_messages,
    ):
        """处理“输入 6 位代码”的一次完整计算：拉行情、算指标、抓新闻、组 prompt、调模型。"""
        quotes_df = _fetch_quotes(symbol_6digit, lookback_days)
        latest_md = _format_recent_quotes(quotes_df, n=recent_rows)

        indicator_summary = ""
        if quotes_df is not None and not getattr(quotes_df, "empty", True):
            indicator_summary = _compute_indicator_summary(quotes_df, lookback_days)

        news_note = ""
        news_md = ""
        if news_rows > 0:
            news_df = _fetch_related_news(symbol_6digit)
            news_md = _format_recent_news(news_df, n=news_rows)
            if news_df is None or getattr(news_df, "empty", True):
                news_note = "新闻接口返回空数据（可能该代码暂无可用新闻或接口变更）"

        prompt = None
        if quotes_df is None or getattr(quotes_df, "empty", True):
            ai_response = (
                "❌ 未获取到行情数据：该代码可能不是 A 股/ETF，或接口临时不可用。"
            )
        else:
            prompt_parts: List[str] = []
            prompt_parts.append(
                f"这是标的代码 {symbol_6digit}（可能是 ETF 或 A 股股票）的最近{recent_rows}行日线行情（注意：仅展示最近N行，但指标按回看{lookback_days}天计算；该表格内容将传入 ETFBot）：\n{latest_md}"
            )
            if indicator_summary:
                prompt_parts.append(
                    f"\n指标摘要（按回看{lookback_days}天计算；将传入 ETFBot）：\n{indicator_summary}"
                )
            if news_rows > 0 and news_md and "未获取" not in news_md:
                prompt_parts.append(
                    f"\n相关新闻摘要（近{news_rows}条；将传入 ETFBot）：\n{news_md}"
                )
            prompt_parts.append(
                "\n提示：ETFBot 的分析与输出风格以 ETF/指数类为主要场景；若该代码为个股，请更谨慎对待新闻/情绪与隔夜风险，并给出更保守的仓位与止损。"
            )
            prompt_parts.append(
                "\n请按‘收盘后复盘 + 开盘前计划/预测（以条件单方式）’的短线波段风格给出可执行计划。"
            )
            prompt = "\n".join(prompt_parts)

            msgs = self._trim_messages_for_llm(
                list(base_messages) + [{"role": "user", "content": prompt}]
            )
            ai_response = self.get_completion_from_messages(msgs)

        return _ComputeResult(
            latest_md=latest_md,
            indicator_summary=indicator_summary,
            news_md=news_md,
            prompt=prompt,
            ai_response=ai_response,
            news_note=news_note,
        )

    def _compute_for_followup(self, user_text: str, base_messages):
        """处理“追问/补充问题”：基于已有上下文直接调用模型。"""
        msgs = self._trim_messages_for_llm(
            list(base_messages) + [{"role": "user", "content": user_text}]
        )
        ai_response = self.get_completion_from_messages(msgs)
        return {"ai_response": ai_response}

    async def handle_send(self, _):
        """处理“发送”：将 UI 输入分流为【代码】或【追问】，并异步在后台线程执行耗时逻辑。"""
        user_input = (self.inp.value or "").strip()
        self.inp.value = ""
        self.status_pane.object = self._render_status()

        if not user_input:
            ai_response = "请提供有效的股票/ETF代码（6位）或相关问题。"
            self.panels.append(
                pn.Row("🤖 ETFBot：", pn.pane.Markdown(ai_response, width=600))
            )
            self._refresh_view()
            return

        is_code = user_input.isdigit() and len(user_input) == 6

        self.panels.append(pn.Row("👤 用户：", pn.pane.Markdown(user_input, width=600)))
        self._refresh_view()

        if (not is_code) and (self.current_symbol is None):
            ai_response = "请先输入 6 位股票/ETF 代码开始会话，然后再提问。"
            self.panels.append(
                pn.Row("🤖 ETFBot：", pn.pane.Markdown(ai_response, width=600))
            )
            self._refresh_view()
            return

        if is_code:
            if self.current_symbol is None:
                self.current_symbol = user_input
            elif self.current_symbol != user_input:
                ai_response = (
                    f"当前会话标的是 {self.current_symbol}。\n\n"
                    "如需切换到新的代码，请先点击【重置会话】，再输入新代码。"
                )
                self.panels.append(
                    pn.Row("🤖 ETFBot：", pn.pane.Markdown(ai_response, width=600))
                )
                self._refresh_view()
                return

        self.active_request_id += 1
        req_id = self.active_request_id

        self._set_processing(True)
        try:
            base_messages = [
                dict(m)
                for m in (
                    self.context or [{"role": "system", "content": SYSTEM_PROMPT}]
                )
            ]

            if is_code:
                lookback_days = int(self.lookback_days_input.value)
                recent_rows = int(self.recent_rows_input.value)
                news_rows = int(self.news_rows_input.value)

                result: _ComputeResult = await self._run_in_thread(
                    self._compute_for_code,
                    self.current_symbol,
                    lookback_days,
                    recent_rows,
                    news_rows,
                    base_messages,
                )

                if req_id != self.active_request_id:
                    return

                self.panels.append(
                    pn.Row(
                        "📈 行情数据（仅展示最近 N 行；这部分将传入 ETFBot）：",
                        pn.pane.Markdown(
                            f"```markdown\n{result.latest_md}\n```", width=600
                        ),
                    )
                )

                if result.indicator_summary:
                    self.panels.append(
                        pn.Row(
                            "📌 指标摘要（用于收盘复盘/开盘前计划；将传入 ETFBot）：",
                            pn.pane.Markdown(result.indicator_summary, width=600),
                        )
                    )

                if news_rows > 0:
                    self.panels.append(
                        pn.Row(
                            "📰 相关新闻（可选；将传入 ETFBot）：",
                            pn.pane.Markdown(
                                result.news_md or "（未获取到相关新闻）", width=600
                            ),
                        )
                    )
                    if (
                        result.news_md and "未获取到" in result.news_md
                    ) and result.news_note:
                        self.panels.append(
                            pn.Row(
                                "ℹ️ 新闻说明：",
                                pn.pane.Markdown(result.news_note, width=600),
                            )
                        )

                self.panels.append(
                    pn.Row(
                        "🤖 ETFBot：", pn.pane.Markdown(result.ai_response, width=600)
                    )
                )

                if result.prompt:
                    self.context.append({"role": "user", "content": result.prompt})
                self.context.append(
                    {"role": "assistant", "content": result.ai_response}
                )
                self.trim_context()

            else:
                result = await self._run_in_thread(
                    self._compute_for_followup, user_input, base_messages
                )
                if req_id != self.active_request_id:
                    return

                ai_response = result.get("ai_response") or ""
                self.panels.append(
                    pn.Row("🤖 ETFBot：", pn.pane.Markdown(ai_response, width=600))
                )
                self.context.append({"role": "user", "content": user_input})
                self.context.append({"role": "assistant", "content": ai_response})
                self.trim_context()

            self._refresh_view()

        except Exception as e:
            if req_id != self.active_request_id:
                return
            ai_response = f"❌ 处理时出错：{type(e).__name__}: {e}"
            self.panels.append(
                pn.Row("🤖 ETFBot：", pn.pane.Markdown(ai_response, width=600))
            )
            self._refresh_view()
        finally:
            if req_id == self.active_request_id:
                self._set_processing(False)

    def _schedule_send(self, event) -> None:
        """Panel 回调入口：用 asyncio.create_task 调度异步发送逻辑。"""
        asyncio.create_task(self.handle_send(event))


def build_dashboard() -> pn.Column:
    """构建 Panel dashboard（返回的对象可直接 `.servable()`）。"""
    app = ETFBotApp()
    return app.dashboard
