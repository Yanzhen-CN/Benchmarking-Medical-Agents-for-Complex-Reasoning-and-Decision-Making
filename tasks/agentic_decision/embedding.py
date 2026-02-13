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
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import os
from config import AgentTaskConfig
# =========================
# memory item 转成 embedding 文本（稳定、可检索）——原样保留
# =========================




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


def _store_done_on_disk(store_root: Path, pid: str, build_note: bool, build_event: bool) -> bool:
    """
    兜底：如果 report 丢了，也能通过磁盘判断是否已经构建完成。
    判定标准：对应 index.faiss 存在（note/event 按开关）。
    """
    base = store_root / pid
    ok = True
    if build_note:
        ok = ok and (base / "note" / "index.faiss").exists()
    if build_event:
        ok = ok and (base / "event" / "index.faiss").exists()
    return ok


if __name__ == "__main__":
    cfg = AgentTaskConfig()

    # -------------------------
    # 1) 确定要处理的 patient 列表
    # -------------------------
    # 推荐：仍然沿用 QUESTIONS_DIR 里的 pid 列表（与你跑 context 的一致）
    files = list(iter_jsonl_files(cfg.QUESTIONS_DIR))
    pids = sorted(set([f.name.split(".")[0] for f in files]))
    if not pids:
        logger.error(f"No pid found from QUESTIONS_DIR={cfg.QUESTIONS_DIR}")
        raise SystemExit(1)

    # -------------------------
    # 2) pre-scan stats + sort
    # -------------------------
    pid_stats: List[Tuple[str, dict]] = []
    for pid in tqdm(pids, desc="Scanning patient sizes"):
        p_path = Path(cfg.PATIENTS_DIR) / f"{pid}.json"
        st = patient_event_stats(p_path)
        pid_stats.append((pid, st))

    pid_stats.sort(
        key=lambda x: (x[1].get("total_events", 0), x[1].get("file_size_bytes", 0)),
        reverse=True,
    )

    if cfg.DEMO_MODE:
        top_k = cfg.DEMO_N
        top = pid_stats[:top_k]
        logger.info("Top patients by event_stream length (total_events desc, file_size desc):")
        for rank, (pid, st) in enumerate(top, 1):
            logger.info(
                f"[{rank}] {pid} | total_events={st.get('total_events')} | "
                f"max_visit_events={st.get('max_visit_events')} | visits={st.get('num_visits')} | "
                f"file_size={st.get('file_size_bytes', 0) / (1024*1024):.2f} MB"
            )
    else:
        top = pid_stats
        logger.warning("DEMO_MODE is OFF, will process ALL patients sorted by size. This may take a long time!")

    # -------------------------
    # 3) resume report
    # -------------------------
    log_dir = Path("log")
    log_dir.mkdir(parents=True, exist_ok=True)

    final_report_name = "final_report_vector_store.json"
    report_path = log_dir / final_report_name

    final_report: Dict[str, Any] = {}
    already_run = set()

    if report_path.exists():
        try:
            previous = json.loads(report_path.read_text(encoding="utf-8"))
            if isinstance(previous, dict):
                final_report = previous
                already_run = set(previous.keys())
                logger.info(f"Resume enabled: loaded {len(already_run)} finished pids from {report_path}")
        except Exception as e:
            logger.warning(f"Failed to load previous report {report_path}: {e}")

    # -------------------------
    # 4) filter pending（report优先；磁盘存在也跳过）
    # -------------------------
    build_note = True
    build_event = True

    pending: List[Tuple[str, dict]] = []
    for pid, st in top:
        if pid in already_run:
            continue
        # 兜底：磁盘已完成也跳过（避免 report 丢了重复跑）
        if _store_done_on_disk(Path(cfg.VECTOR_STORE_DIR), pid, build_note, build_event):
            # 写一条“磁盘判定跳过”的记录（可选）
            final_report[pid] = {
                "skipped": True,
                "reason": "store_exists_on_disk",
                "stats": st,
            }
            already_run.add(pid)
            continue
        pending.append((pid, st))

    # 立即落盘一次（把 disk-skip 写进去）
    report_path.write_text(json.dumps(final_report, ensure_ascii=False, indent=2), encoding="utf-8")

    if not pending:
        logger.info("Nothing to do: all selected patients already processed (report or disk).")
        logger.info(f"Existing report: {report_path}")
        raise SystemExit(0)

    logger.info(f"Will process {len(pending)} patients (skipping {len(top) - len(pending)} already-run).")

    # -------------------------
    # 5) token usage accumulate
    # -------------------------
    usage = {
        "embeddings": {
            "input_tokens": 0,
            "total_tokens": 0,
        }
    }

    # -------------------------
    # 6) worker
    # -------------------------
    def _run_one(pid: str, st: dict) -> Tuple[str, dict]:
        """
        每个 pid：
          - 读取 patient.json + sequenced.json
          - 写 vector_store/{pid}/note, event
          - 返回一个 result dict 供主线程记报告
        """
        patients_dir = Path(cfg.PATIENTS_DIR)
        sequence_dir = Path(cfg.EVENT_SEQ_DIR)
        store_root = Path(cfg.VECTOR_STORE_DIR)

        patient_json = patients_dir / f"{pid}.json"
        sequenced_json = sequence_dir / f"{pid}_sequenced.json"

        if not patient_json.exists():
            raise FileNotFoundError(f"missing patient_json: {patient_json}")
        if not sequenced_json.exists():
            raise FileNotFoundError(f"missing sequenced_json: {sequenced_json}")

        t0 = time.time()
        token_usage = build_patient_global_stores(
            patient_json_path=patient_json,
            sequenced_json_path=sequenced_json,
            out_root=store_root,
            emb_model=cfg.EMBEDDING_MODEL,
            batch_size=getattr(cfg, "BATCH_SIZE", 8),
            normalize=getattr(cfg, "NORMALIZE_EMBEDDINGS", True),
            build_note=build_note,
            build_event=build_event,
        )
        dt = time.time() - t0

        # 统一返回结构：便于 resume / debug
        result = {
            "ok": True,
            "seconds": dt,
            "stats": st,
            "store_root": str((store_root / pid).resolve()),
            "build_note": build_note,
            "build_event": build_event,
            "token_usage": token_usage,  # llm.get_token_usage()
        }
        return pid, result

    # -------------------------
    # 7) concurrency
    # -------------------------
    max_workers = getattr(cfg, "MAX_WORKERS", None)
    if not max_workers:
        max_workers = min(8, (os.cpu_count() or 8))
    logger.info(f"Running with max_workers={max_workers}")

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_run_one, pid, st): (pid, st) for pid, st in pending}

        for fut in tqdm(as_completed(futs), total=len(futs), desc="Building vector stores (concurrent)"):
            pid, st = futs[fut]
            try:
                pid_done, result = fut.result()
            except Exception as e:
                logger.exception(f"FAILED pid={pid}: {e}")
                final_report[pid] = {
                    "ok": False,
                    "error": str(e),
                    "stats": st,
                }
            else:
                final_report[pid_done] = result
                # accumulate embeddings usage (best-effort)
                tu = (result.get("token_usage") or {}).get("embeddings") or {}
                usage["embeddings"]["input_tokens"] += int(tu.get("input_tokens", 0) or 0)
                usage["embeddings"]["total_tokens"] += int(tu.get("total_tokens", 0) or 0)

                logger.info(
                    f"Done pid={pid_done}: store={result.get('store_root')} "
                    f"emb_tokens={tu.get('total_tokens', 0)} time={result.get('seconds', 0):.1f}s"
                )

            # incremental save after each completion
            report_path.write_text(json.dumps(final_report, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("All done (concurrent).")
    logger.info(f"Total embedding token usage across processed patients:\n{json.dumps(usage, ensure_ascii=False, indent=2)}")
    logger.info(f"Final report: {report_path}")