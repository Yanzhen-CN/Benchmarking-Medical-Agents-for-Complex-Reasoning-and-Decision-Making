# mem0_agent (黑箱记忆 Agent)

本目录提供一个**独立配置的 mem0 记忆 agent**（黑箱接口），用于在 benchmark 中按“对话输入 → 可选检索 → 生成 → 写入记忆”的方式运行。

- 入口脚本：`run.py`
- 独立配置：`./.env`（只在本目录生效）
- 记忆后端：mem0（可替换）
- LLM 后端：OpenAI-compatible（当前配 qwen-turbo）

---

## 1) 运行方式

```bash
cd /data/zxc/Benchmarking-Medical-Agents-for-Complex-Reasoning-and-Decision-Making/agents/mem0_agent
python run.py
```

> `run.py` 会**只加载本目录的 `.env`**，不会影响仓库根目录的环境变量。

### run_test（测试序列任务）
```bash
python run_test.py --task test --items P0001 --debug
```

输出会写入到：
```
run/<task>/<run_id>/<patient_id>.jsonl
```

其中 `run_id` 为 `YYYYMMDD-HHMMSS-<8位随机>`。

更多示例：
```bash
# 运行单个患者，开启 debug
python run_test.py --task tests --items P0001 --debug

# 运行多个患者（逗号分隔）
python run_test.py --task tests --items P0001,P0002

# 运行某个 task 下的所有 patients
python run_test.py --task tests --items all

# 保留记忆（不做删除）
python run_test.py --task tests --items P0001 --no-delete
```

> `MEM0_INDEX_WAIT_S` 在 `.env` 里配置；运行时不需要再传。

---

## 2) Agent 总体逻辑

**输入**：一组对话消息 `messages`（chat completions 格式）

**处理流程**：
1. 取最后一条 user 消息作为 query
2. 根据 `AGENT_RETRIEVAL_POLICY` 判断是否检索
3. 若需要检索：调用 mem0 `search` 获取 top‑k 记忆
4. 拼接 system prompt + 检索结果 + 最近 N 轮对话
5. 调用 LLM 生成回答
6. 写入记忆：
   - 对话原文（user + assistant）
   - 可选：抽取 observations 再写入

**输出**：assistant 回复字符串

---

## 3) 参数说明（.env）

### LLM 相关
- `OPENAI_API_KEY`
  - OpenAI-compatible API key（qwen 可用）
- `OPENAI_API_BASE_URL`
  - OpenAI-compatible base URL（qwen 的 compatible endpoint）
- `LLM_MODEL`
  - chat completions 使用的模型名（例：`qwen-turbo`）

### mem0 相关
- `MEM0_API_KEY`
  - mem0 API Key（必须）
- `MEM0_ORG_ID`
  - 可选：组织隔离
- `MEM0_PROJECT_ID`
  - 可选：项目隔离
- `MEM0_SYNC_WRITE`
  - 记忆写入是否同步（1=同步，0=异步）。建议保持 1，避免删不干净
- `MEM0_PRINT_IDS`
  - 调试开关：打印写入返回的 memory_id（1=打印，0=关闭）
- `MEM0_INDEX_WAIT_S`
  - 每个问题前等待秒数（用于索引延迟，默认 0）

### Agent identity / scoping（隔离用）
- `AGENT_ID`
  - agent 标识（建议固定，如 `bench-agent`）
- `AGENT_APP_ID`
  - 应用标识（建议固定，如 `medagentbench`）
- `AGENT_RUN_ID`
  - 任务/实验 run 的标识（可用于 task 级隔离）

> **隔离建议**：
> - `user_id = patient_id`
> - `run_id = 每次运行的唯一标识`
> - `agent_id = bench-agent`
> - `app_id = medagentbench`

> 说明：`run_test.py` 会自动生成 `run_id=YYYYMMDD-HHMMSS-<8位随机>`，不会读取 `AGENT_RUN_ID`。

### Retrieval 行为
- `MEMORY_TOP_K`
  - 记忆检索返回的条数（top‑k）
- `AGENT_MAX_RECENT_TURNS`
  - prompt 中保留的最近对话轮数
- `AGENT_INCLUDE_MEMORY`
  - 是否把检索到的记忆放进 prompt（1/0）
- `AGENT_RETRIEVAL_POLICY`
  - 检索策略：
    - `always`：每轮都检索
    - `question_only`：仅问句检索（默认推荐）
    - `never`：永不检索（只写入）

### 写入行为
- `AGENT_STORE_DIALOG`
  - 是否写入原文对话（user+assistant）
- `AGENT_STORE_OBS`
  - 是否写入 observation（LLM 抽取的事实）

---

## 4) 输入格式（run_test）

读取路径固定为 `question_data/<task>/<patient_id>.jsonl`。

每个患者文件为 JSONL（兼容 `.jsonl` / `.json` 扩展名），每行一个事件：

```json
{"id": 0, "type": "fact", "data": "患者三天前开始发热并咳嗽。"}
{"id": 1, "type": "fact", "data": "既往有2型糖尿病和高血压。"}
{"id": 2, "type": "question", "data": "患者最高体温是多少？"}
```

字段说明：
- `id`: 事件编号
- `type`: `fact` 或 `question`
- `data`: 文本内容

---

## 5) 输出格式（run_test）

输出为每个患者一个 JSONL 文件：

```
run/<task>/<run_id>/<patient_id>.jsonl
```

内容为问答结果，每行一条：

```json
{"id": 3, "answer": "患者最高体温是39.2°C。"}
{"id": 5, "answer": "患者有高血压和糖尿病的基础疾病。"}
```
