# retrieval/build_vector_store_from_memories.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import faiss  # pip install faiss-cpu

from util.llmUtil import LLMUtil
from tasks.agentic_decision.tools import *  # parse_qid 等
from util.logUtil import setup_logger
logger = setup_logger()
from tqdm import tqdm

# =========================
# memory item 转成 embedding 文本（稳定、可检索）——原样保留
# =========================

def _safe_str(x: Any) -> str:
    return "" if x is None else str(x).strip()

def _chunk_text(text: str, max_chars: int = 1200, overlap: int = 150) -> List[str]:
    t = _safe_str(text)
    if not t:
        return []
    if len(t) <= max_chars:
        return [t]
    out = []
    i = 0
    while i < len(t):
        j = min(len(t), i + max_chars)
        out.append(t[i:j])
        if j == len(t):
            break
        i = max(0, j - overlap)
    return out

def memory_item_to_docs(patient_id: str, item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    把 note/event memory item 变成若干条 doc（text+meta）
    NOTE: 这里不做可视范围约束；可视范围过滤放到检索阶段。
    """
    mtype = item.get("memory_type")
    docs: List[Dict[str, Any]] = []

    if mtype == "note":
        text = _safe_str(item.get("text"))
        if not text:
            return []
        meta = {
            "patient_id": patient_id,
            "memory_type": "note",
            "visit_id": item.get("visit_id"),
            "visit_ref": item.get("visit_ref"),   # ✅ 新增：可选
            "visit_idx": item.get("visit_idx"),   # ✅ 新增：可选
            "note_type": item.get("note_type"),
        }
        docs.append({"text": text, "meta": meta})
        return docs

    if mtype != "event":
        return []

    et = _safe_str(item.get("event_type"))
    ts = item.get("timestamp")
    vref = _safe_str(item.get("visit_ref"))
    eid = _safe_str(item.get("event_id"))
    content = item.get("content") or {}

    base_meta = {
        "patient_id": patient_id,
        "memory_type": "event",
        "visit_ref": vref,
        "visit_idx": item.get("visit_idx"),  # ✅ 新增：可选（我们会补上）
        "event_id": eid,
        "event_type": et,
        "timestamp": ts,
    }

    if et == "LAB":
        for idx, x in enumerate(content.get("items") or []):
            text = (
                f"[LAB] t={ts} visit={vref} | "
                f"name={_safe_str(x.get('name'))} | "
                f"value={_safe_str(x.get('value_text') or x.get('value_num'))} {_safe_str(x.get('unit'))} | "
                f"flag={_safe_str(x.get('flag'))} | fluid={_safe_str(x.get('fluid'))} | "
                f"category={_safe_str(x.get('category'))}"
            )
            meta = dict(base_meta)
            meta["item_index"] = idx
            docs.append({"text": text, "meta": meta})
        return docs

    if et == "MEDICATION":
        for idx, x in enumerate(content.get("items") or []):
            text = (
                f"[MED] t={ts} visit={vref} | "
                f"drug={_safe_str(x.get('drug'))} | route={_safe_str(x.get('route'))} | "
                f"dose={_safe_str(x.get('dose'))} | status={_safe_str(x.get('status'))} | "
                f"end={_safe_str(x.get('end_timestamp'))}"
            )
            meta = dict(base_meta)
            meta["item_index"] = idx
            docs.append({"text": text, "meta": meta})
        return docs

    if et == "MICROBIOLOGY":
        specimen = content.get("specimen") or {}
        results = content.get("results") or {}
        comments = results.get("comments") or []
        text = (
            f"[MICRO] t={ts} visit={vref} | "
            f"spec_type={_safe_str(specimen.get('spec_type'))} | test={_safe_str(specimen.get('test_name'))} | "
            f"negative={_safe_str(results.get('negative'))} | "
            f"organisms={_safe_str(results.get('organisms'))} | "
            f"comments={' '.join(map(_safe_str, comments))}"
        )
        docs.append({"text": text, "meta": base_meta})
        return docs

    if et == "IMAGING":
        report = _safe_str(content.get("report"))
        chunks = _chunk_text(report, max_chars=1200, overlap=150)
        for k, ch in enumerate(chunks):
            meta = dict(base_meta)
            meta["chunk_id"] = k
            docs.append({"text": f"[IMAGING] t={ts} visit={vref} | chunk={k+1}/{len(chunks)}: {ch}", "meta": meta})
        return docs

    if et == "PROCEDURE":
        items = content.get("items") or []
        text = (
            f"[PROC] t={ts} visit={vref} | "
            f"items={'; '.join(_safe_str(x.get('name')) for x in items)}"
        )
        docs.append({"text": text, "meta": base_meta})
        return docs

    brief = _safe_str(json.dumps(content, ensure_ascii=False))
    if brief:
        docs.append({"text": f"[{et}] t={ts} visit={vref} | {brief[:2000]}", "meta": base_meta})
    return docs


# =========================
# 向量化 + 落盘（改：只按 patient 存一份）
# =========================

def _l2_normalize_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / n

def build_store(
    *,
    llm: LLMUtil,
    docs: List[Dict[str, Any]],
    out_dir: Path,
    emb_model: str,
    batch_size: int = 8,
    max_text_chars: int = 6000,
    normalize: bool = True,
) -> None:
    texts: List[str] = []
    for d in docs:
        t = d.get("text", "")
        t = t if isinstance(t, str) else str(t)
        if len(t) > max_text_chars:
            t = t[:max_text_chars]
        d["text"] = t
        texts.append(t)

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) save docs
    with (out_dir / "docs.jsonl").open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    if not texts:
        np.save(out_dir / "emb.npy", np.zeros((0, 1), dtype=np.float32))
        faiss.write_index(faiss.IndexFlatIP(1), str(out_dir / "index.faiss"))
        logger.info(f"[EMPTY] store written: {out_dir}")
        return

    # 2) embed (in batches)
    vecs: List[List[float]] = []
    for i in tqdm(range(0, len(texts), batch_size)):
        buf = texts[i:i + batch_size]
        vecs.extend(llm.embed_texts(buf, model=emb_model, batch_size=len(buf)))

    embs = np.asarray(vecs, dtype=np.float32)
    if normalize:
        embs = _l2_normalize_rows(embs)

    np.save(out_dir / "emb.npy", embs)

    # 3) index
    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim) if normalize else faiss.IndexFlatL2(dim)
    index.add(embs)
    faiss.write_index(index, str(out_dir / "index.faiss"))

    logger.info(f"[OK] store={out_dir} docs={len(docs)} dim={dim}")


# =========================
# 新增：构建“全量 message”的 memory items（不截断）
# =========================

def _note_dict_to_text(note: Any) -> str:
    if note is None:
        return ""
    if isinstance(note, str):
        return note.strip()
    if isinstance(note, dict):
        parts = []
        for k, v in note.items():
            if v is None:
                continue
            vv = v.strip() if isinstance(v, str) else str(v)
            if not vv:
                continue
            parts.append(f"{k}: {vv}")
        return "\n".join(parts).strip()
    return str(note).strip()

def build_all_note_items(patient_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    ✅ 全量：遍历所有 visits 的 admission_note/discharge_note
    输出为 memory items（memory_type=note），后续复用 memory_item_to_docs()。
    """
    visits = patient_obj.get("visits") or []
    items: List[Dict[str, Any]] = []
    for i, v in enumerate(visits):
        vid = v.get("visit_id")
        if not isinstance(vid, str):
            continue
        vref = None
        if "-V" in vid:
            vref = "V" + vid.split("-V")[1]

        adm = (((v.get("admission_info") or {}).get("admission_note")))
        dis = (((v.get("discharge_info") or {}).get("discharge_note")))

        adm_text = _note_dict_to_text(adm)
        dis_text = _note_dict_to_text(dis)

        if adm_text:
            items.append({
                "memory_type": "note",
                "visit_id": vid,
                "visit_ref": vref,
                "visit_idx": i,
                "note_type": "admission_note",
                "text": adm_text,
            })
        if dis_text:
            items.append({
                "memory_type": "note",
                "visit_id": vid,
                "visit_ref": vref,
                "visit_idx": i,
                "note_type": "discharge_note",
                "text": dis_text,
            })
    return items

def build_all_event_items(
    seq_events: List[Dict[str, Any]],
    visit_order: Dict[str, int],
) -> List[Dict[str, Any]]:
    """
    ✅ 全量：遍历 sequenced_json 的所有事件
    给每条事件补 visit_idx（来自 visit_order），方便检索后过滤。
    输出为 memory items（memory_type=event），后续复用 memory_item_to_docs()。
    """
    items: List[Dict[str, Any]] = []
    for ev in tqdm(seq_events):
        vref = ev.get("visit_ref")
        if not vref or vref not in visit_order:
            # 没有 visit_ref 的事件你也可以选择保留（visit_idx=None），这里默认跳过
            continue
        items.append({
            "memory_type": "event",
            "visit_ref": vref,
            "visit_idx": visit_order[vref],
            "event_id": ev.get("event_id"),
            "event_type": ev.get("event_type"),
            "timestamp": ev.get("timestamp"),
            "content": ev.get("content"),
        })
    return items


def build_patient_global_stores(
    *,
    patient_json_path: Path,
    sequenced_json_path: Path,
    out_root: Path,
    emb_model: str = "text-embedding-v4",
    batch_size: int = 8,
    normalize: bool = True,
    build_note: bool = True,
    build_event: bool = True,
) -> Dict[str, Any]:
    """
    ✅ 改动点：每个 patient 只建一份全量索引
    out:
      vector_store/{patient_id}/note/index.faiss
      vector_store/{patient_id}/event/index.faiss
    """
    patient_obj = json.loads(patient_json_path.read_text(encoding="utf-8"))
    patient_id = patient_json_path.stem

    visits = patient_obj.get("visits") or []

    # visit_order: {"V1":0, "V2":1, ...}
    visit_order: Dict[str, int] = {}
    for i, v in enumerate(visits):
        vid = v.get("visit_id")
        if not isinstance(vid, str) or "-V" not in vid:
            continue
        vref = "V" + vid.split("-V")[1]
        visit_order[vref] = i

    seq_events = json.loads(sequenced_json_path.read_text(encoding="utf-8"))
    if not isinstance(seq_events, list):
        raise ValueError(f"sequenced json must be list: {sequenced_json_path}")

    llm = LLMUtil()
    base = out_root / patient_id

    if build_note:
        note_items = build_all_note_items(patient_obj)
        note_docs: List[Dict[str, Any]] = []
        for it in note_items:
            note_docs.extend(memory_item_to_docs(patient_id, it))
        logger.info(f"patient={patient_id} note_items={len(note_items)} -> note_docs={len(note_docs)}")
        build_store(
            llm=llm,
            docs=note_docs,
            out_dir=base / "note",
            emb_model=emb_model,
            batch_size=batch_size,
            normalize=normalize,
        )

    if build_event:
        event_items = build_all_event_items(seq_events, visit_order=visit_order)
        event_docs: List[Dict[str, Any]] = []
        for it in event_items:
            event_docs.extend(memory_item_to_docs(patient_id, it))
        logger.info(f"patient={patient_id} event_items={len(event_items)} -> event_docs={len(event_docs)}")
        build_store(
            llm=llm,
            docs=event_docs,
            out_dir=base / "event",
            emb_model=emb_model,
            batch_size=batch_size,
            normalize=normalize,
        )

    logger.info(f"[DONE] patient={patient_id} note={build_note} event={build_event} -> {base}")
    return llm.get_token_usage()

from config import AgentTaskConfig

if __name__ == "__main__":
    patient_id = "P000001"
    cfg = AgentTaskConfig()
    patients_dir = cfg.PATIENTS_DIR
    sequence_dir = cfg.EVENT_SEQ_DIR
    vector_store_dir = cfg.VECTOR_STORE_DIR
    usage = {
        "input_tokens": 0,
        "total_tokens": 0,
    }
    
    result = build_patient_global_stores(
        patient_json_path=Path(f"{patients_dir}/{patient_id}.json"),
        sequenced_json_path=Path(f"{sequence_dir}/{patient_id}_sequenced.json"),
        out_root=Path(vector_store_dir),
        emb_model=cfg.EMBEDDING_MODEL,
        batch_size=cfg.BATCH_SIZE,
        normalize=cfg.NORMALIZE_EMBEDDINGS,
        build_note=True,
        build_event=True,
    )
    usage["input_tokens"] += result["embeddings"]["input_tokens"]
    usage["total_tokens"] += result["embeddings"]["total_tokens"]
    logger.info(f"embedding token usage: {json.dumps(usage, indent=2)}")
