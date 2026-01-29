from typing import Dict, Any, Tuple, List, Optional
import json
import re

from util.llmUtil import LLMUtil
from util.logUtil import setup_logger
logger = setup_logger()
from config import ContextConfig, LLMConfig
context_config = ContextConfig()
llm_config = context_config.llm_config


if context_config.USE_LLM_FOR_IMAGE_DESC or context_config.USE_LLM_FOR_REASON:
    llm = LLMUtil()

def infer_imaging_modality_target_llm(
    report_text: str,
    model: str = "qwen-turbo",
) -> Tuple[str, str, float]:
    """
    LLM-only classification. Returns (modality, target, confidence).
    Must NOT produce findings.
    Fallback: caller should use heuristic if confidence low or invalid.
    """
    
    if not context_config.USE_LLM_FOR_IMAGE_DESC:
        logger.error("LLM usage for imaging modality/target inference is disabled in config.")
        return "", "", 0.0
    
    system_prompt = (
        "You are a clinical text classifier.\n"
        "Task: classify imaging modality and anatomical target from the given text.\n"
        "Constraints:\n"
        "- Output ONLY JSON with keys: modality, target, confidence, evidence_phrases.\n"
        "- modality and target must copied from the text. If not, output empty string for modality and target, and 0.0 for confidence.\n"
        "- Do NOT include any imaging findings, interpretations, or diagnosis.\n"
        "- evidence_phrases must be short substrings copied from input.\n"
    )

    obj = llm.chat_json(
        system_prompt=system_prompt,
        user_text=report_text[:2000],
        model=model,
        temperature=1.0,
        max_tokens=256,
        strict_only_json=True
    )

    modality = str(obj.get("modality") or "").strip()
    target = str(obj.get("target") or "").strip()
    confidence = float(obj.get("confidence") or 0.0)

    # evidence check: phrases must appear in original text (case-insensitive)
    evidence_phrases = obj.get("evidence_phrases") or []
    ok_evidence = True
    if isinstance(evidence_phrases, list) and evidence_phrases:
        low = report_text.lower()
        for ph in evidence_phrases[:5]:
            if not isinstance(ph, str):
                ok_evidence = False
                break
            if ph.strip().lower() not in low:
                ok_evidence = False
                break
            
    if not ok_evidence:
        # still return but with reduced confidence
        confidence = min(confidence, 0.3)

    return modality, target, confidence

import json
from typing import Any, Dict, List, Optional

def generate_reason_from_messages_llm(
    messages_so_far: List[Dict[str, Any]],
    current_action: str,
    current_args: Dict[str, Any],
    model: str = "qwen-turbo",
    max_history_msgs: int = 24,   # 控制上下文长度：只取最近 N 条
) -> str:
    """
    Generate assistant 'reason' based ONLY on the prefix messages (no extra summary).
    Returns "" if fails/unsafe; caller should fallback to template reason.
    """

    if not context_config.USE_LLM_FOR_REASON:
        logger.debug("LLM usage for reason generation is disabled in config.")
        return ""
    # 只给最近N条，避免上下文过长；保留所有system也可以，但通常2条system + 最近历史足够
    # 这里做一个简单策略：保留所有 system + 最近 (max_history_msgs) 条非-system
    sys_msgs = [m for m in messages_so_far if m.get("role") == "system"]
    non_sys = [m for m in messages_so_far if m.get("role") != "system"]
    clipped = sys_msgs + non_sys[-max_history_msgs:]

    system_prompt = (
        "You write the 'reason' field for the next assistant action in a doctor-agent dialogue.\n"
        "Hard constraints:\n"
        "- You may ONLY use information present in the provided prefix_messages.\n"
        "- Do NOT invent new patient facts, findings, diagnoses, or outcomes.\n"
        "- Do NOT mention results of the current action (results are not available yet).\n"
        "- Keep it 1-2 sentences, not exceeding 50 words, workflow-rationale focused.\n"
        "Output ONLY JSON: {\"reason\": string, \"constraints_ok\": boolean}.\n"
    )

    user_text = json.dumps({
        "prefix_messages": clipped,
        "current_decision": {
            "action": current_action,
            "args": current_args
        }
    }, ensure_ascii=False)

    obj = llm.chat_json(
        system_prompt=system_prompt,
        user_text=user_text,
        model=model,
        temperature=0.0,
        strict_only_json=True
    )

    if not isinstance(obj, dict) or obj.get("constraints_ok") is not True:
        return ""

    reason = obj.get("reason")
    if not isinstance(reason, str):
        return ""

    reason = reason.strip()
    if not reason:
        return ""
    logger.debug(f"Generated reason from LLM: {reason}")
    # # 轻量防“提前知道结果”的措辞（可以按你们数据集再扩展）
    # banned = ["revealed", "showed", "confirmed", "consistent with", "positive for", "negative for", "rules out"]
    # if any(b in reason.lower() for b in banned):
    #     return ""

    # 限长
    if len(reason) > 1024:
        reason = reason[:1024].rstrip() + "..."

    return reason
