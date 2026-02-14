import json
from typing import Any, List, Optional, Tuple, Set, Dict
from pathlib import Path

def _parse_json_answer(reply: Any) -> Tuple[Optional[List[str]], Optional[str], str]:
    """
    return: (answer_list or None, reason or None, raw_text)
    """
    raw = reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)
    raw = raw.strip()

    # 兼容模型输出 ```json ... ``` 的情况
    if raw.startswith("```"):
        raw = raw.strip("`")
        # 可能是 "json\n{...}" 这种
        raw = raw.split("\n", 1)[-1].strip()

    try:
        obj = json.loads(raw)
    except Exception:
        return None, None, raw

    if not isinstance(obj, dict):
        return None, None, raw

    reason = obj.get("reason")
    ans = obj.get("answer")

    if isinstance(ans, list):
        ans_list = []
        for x in ans:
            if isinstance(x, str):
                s = x.strip()
                if s:
                    ans_list.append(s)
        return (ans_list if ans_list else None), (reason if isinstance(reason, str) else None), raw

    # 允许模型偶尔给 str：转成单元素 list
    if isinstance(ans, str) and ans.strip():
        return [ans.strip()], (reason if isinstance(reason, str) else None), raw

    return None, (reason if isinstance(reason, str) else None), raw

from typing import Any, List, Set

def _normalize_list(x: List[str]) -> List[str]:
    return [s.strip() for s in x if isinstance(s, str) and s.strip()]


def score_weighted_acc(gt_answer: Any, pred_list: List[str]) -> float:
    """
    支持：
      - gt_answer: str
      - gt_answer: dict {option: weight}
      - gt_answer: list[str]
    pred_list: list[str]（来自 JSON answer）

    规则：
      1) gt 为 str → 单选精确匹配
      2) gt 为 dict → 加权求和（pred ∩ gt）
      3) gt 为 list → 集合匹配（完全正确=1，否则0）
    """

    pred_list = _normalize_list(pred_list)
    if not pred_list:
        return 0.0

    # ---- Case 1: gt 是单个字符串 ----
    if isinstance(gt_answer, str):
        gt = gt_answer.strip()
        # 只要 pred_list 中包含 gt 即判 1
        return 1.0 if gt in pred_list else 0.0

    # ---- Case 2: gt 是加权 dict ----
    if isinstance(gt_answer, dict):
        score = 0.0
        for p in pred_list:
            for gt, w in gt_answer.items():
                if gt.strip().lower() in p.strip().lower():
                    score += float(w)
        # 若权重总和理论最大为 1，可直接返回
        return max(0.0, min(1.0, float(score)))

    # ---- Case 3: gt 是 list[str]（多选无权重）----
    if isinstance(gt_answer, list):
        gt_list = _normalize_list(gt_answer)
        gt_set: Set[str] = set(gt_list)
        pred_set: Set[str] = set(pred_list)

        return len(pred_set & gt_set) / len(gt_set)

    return 0.0

def format_mcq_user_content(q: Dict[str, Any]) -> str:
    """
    把题目+候选项组织成最后一条 user 发言。
    兼容你现有 jsonl：question/options/qtype/...
    """
    question = q.get("question", "")
    options = q.get("options") or []
    if not isinstance(options, list):
        options = []

    lines = [str(question).strip(), "", "Options:"]
    for i, opt in enumerate(options):
        lines.append(f"{opt}")
    lines.append("")
    lines.append("Please answer with the option strings exactly as listed. Output json schema as shown in system prompt.")
    return "\n".join(lines)


_ALLOWED_ROLES = {"system", "user", "assistant", "tool"}

def _to_str_content(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    # dict / list / 其它对象
    try:
        return json.dumps(x, ensure_ascii=False)
    except Exception:
        return str(x)

def normalize_messages(msgs: List[Dict[str, Any]],
                       max_chars: Optional[int] = None) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = m.get("role", "user")
        if role not in _ALLOWED_ROLES:
            # 兜底：未知 role 当成 assistant 或 tool 都行，这里当 tool
            role = "tool"
        out.append(
            {
                "role": role,
                "content": _to_str_content(m.get("content")),
            }
        )
    return out

def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
            
def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)