# ETFBot

一个基于 **Panel** 的本地交互式小工具：输入 6 位 **场内 ETF / A 股代码**，拉取日线行情（AkShare），计算常见技术指标，并通过 **OpenAI-compatible** 接口调用大模型（默认使用 Gemini 兼容端点）生成：**收盘后复盘** + **开盘前计划/预测（条件单思维）**。

> 设计偏 ETF/指数类风格；个股同样可用，但建议更保守并结合公告/事件风险。

## 目录结构

- `etf_bot.ipynb`：Notebook 启动入口（可 `panel serve` 或在 Notebook 中 `servable()`）
- `etfbot_app.py`：Panel 应用主体（取数/指标/新闻/UI 编排与调用 LLM）
- `llm_client.py`：LLM 调用封装（OpenAI-compatible；默认 Gemini 兼容端点；仅从环境变量读取 API Key）
- `README.md`：使用说明

## 环境要求

- Python 3.12（建议）
- 依赖：`panel`、`akshare`、`pandas`、`openai`

## 安装依赖

在你的虚拟环境中执行：

```bash
pip install panel akshare pandas openai
```

> AkShare 可能会有额外依赖（由 pip 自动安装或按提示补齐）。

## 配置 LLM Key（OpenAI-compatible）

默认使用 **OpenAI-compatible** 模式（`/v1/chat/completions`），并默认指向 Gemini 的 OpenAI-compatible 端点。

需要环境变量：`LLM_API_KEY`

兼容环境变量（可不改旧习惯）：`GEMINI_API_KEY` / `OPENAI_API_KEY`

说明：

- `BASE_URL` 与 `DEFAULT_MODEL` **不从环境变量读取**，需要的话请直接改 [llm_client.py](llm_client.py) 里的常量（默认：`gemini-2.5-flash`）。
- 若要切换到其他 OpenAI-compatible 服务：将 `BASE_URL` 改为你的服务地址，并将 `DEFAULT_MODEL` 改为该服务支持的模型名，`LLM_API_KEY` 对应修改即可。

PowerShell：

```powershell
$env:LLM_API_KEY="你的key"
```

CMD：

```bat
set LLM_API_KEY=你的key
```

## 运行方式

### 在浏览器中启动（推荐）

在项目目录执行：

```bash
panel serve etf_bot.ipynb --show
```

> `--show` 会自动打开浏览器；默认地址通常是 http://localhost:5006/etf_bot

1. 打开 `etf_bot.ipynb`
2. 运行第 1 个单元格（会创建并 `servable()` 出 Dashboard）
3. 在界面中输入：
   - 6 位代码：如 `512800`、`000001`
   - 或者在已选定标的后继续追问

## 常见问题

- **没有配置 API Key 会怎样？**
  - 工具仍会尝试拉行情/算指标，但不会调用模型，界面会提示先配置 `LLM_API_KEY`（或兼容变量）。
- **新闻获取为空？**
  - 新闻接口是 best-effort，可能为空或接口变更；不影响行情与指标部分。
