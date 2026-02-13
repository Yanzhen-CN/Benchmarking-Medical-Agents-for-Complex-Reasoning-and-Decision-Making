from __future__ import annotations

import argparse
from pathlib import Path
import json
import os
import sys
import time
import datetime
import uuid
import dotenv

# Ensure repo root is on sys.path for `import agents`
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.mem0_agent import (
    MemoryAugmentedChatAgent,
    OpenAICompatibleLLMProvider,
    Mem0MemoryProvider,
    LLMObservationExtractor,
    InMemoryProvider,
)
from agents.mem0_agent.core import AgentConfig


def load_local_env() -> None:
    env_path = Path(__file__).parent / ".env"
    # Force local env to take precedence over repo/global .env
    dotenv.load_dotenv(env_path, override=True)


def build_agent():
    llm = OpenAICompatibleLLMProvider()

    provider = os.getenv("MEMORY_PROVIDER", "mem0").lower()
    if provider == "in_memory":
        mem = InMemoryProvider()
    else:
        mem = Mem0MemoryProvider()

    obs = LLMObservationExtractor(llm)

    cfg = AgentConfig(
        max_recent_turns=int(os.getenv("AGENT_MAX_RECENT_TURNS", "6")),
        memory_top_k=int(os.getenv("MEMORY_TOP_K", "5")),
        store_dialog=os.getenv("AGENT_STORE_DIALOG", "1") == "1",
        store_observations=os.getenv("AGENT_STORE_OBS", "1") == "1",
        include_memory_in_prompt=os.getenv("AGENT_INCLUDE_MEMORY", "1") == "1",
        retrieval_policy=os.getenv("AGENT_RETRIEVAL_POLICY", "question_only"),
        query_rewrite=os.getenv("AGENT_QUERY_REWRITE", "1") == "1",
    )

    return MemoryAugmentedChatAgent(
        llm=llm,
        memory=mem,
        observation_extractor=obs,
        config=cfg,
    ), mem


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def store_fact(mem, text: str, *, user_id: str, agent_id: str, app_id: str, run_id: str) -> None:
    mem.add_memory(
        text=text,
        metadata={"type": "fact"},
        user_id=user_id,
        agent_id=agent_id,
        app_id=app_id,
        run_id=run_id,
    )


def maybe_flush(mem, *, user_id: str, agent_id: str, app_id: str, run_id: str) -> None:
    # Note: Mem0 writes are synchronous by default when MEM0_SYNC_WRITE=1.
    try:
        mem.delete_all(user_id=user_id, agent_id=agent_id, app_id=app_id, run_id=run_id)
    except Exception as exc:
        print(f"[warn] Memory cleanup failed: {exc}", file=sys.stderr)


def main() -> None:
    load_local_env()

    parser = argparse.ArgumentParser(description="Run mock sequence test")
    parser.add_argument("--task", required=True, help="task name under tasks/<task>/sequence")
    parser.add_argument(
        "--items",
        required=True,
        help="comma-separated list of json basenames (e.g., P0001,P0002) or 'all'",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="print retrieved memories for each question",
    )
    parser.add_argument(
        "--no-delete",
        action="store_true",
        help="skip memory cleanup and print memory ids at the end",
    )
    args = parser.parse_args()

    seq_dir = ROOT / "tasks" / args.task / "sequence"

    if args.items.strip().lower() == "all":
        item_names = [p.stem for p in sorted(seq_dir.glob("*.json"))]
        if not item_names:
            raise SystemExit(f"No items found in {seq_dir}")
    else:
        item_names = [x.strip() for x in args.items.split(",") if x.strip()]
        if not item_names:
            raise SystemExit("No items provided")

    agent, mem = build_agent()

    app_id = os.getenv("AGENT_APP_ID", "medagentbench")
    agent_id = os.getenv("AGENT_ID", "bench-agent")

    # Run-scoped id: <timestamp>-<rand>
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    rid = uuid.uuid4().hex[:8]
    run_id = f"{ts}-{rid}"

    for item in item_names:
        path = seq_dir / f"{item}.json"
        if not path.exists():
            raise SystemExit(f"Missing file: {path}")

        # Memory isolation: run_id uses time + random (per run)
        user_id = item

        answers = []

        for obj in iter_jsonl(path):
            obj_type = obj.get("type")
            obj_id = obj.get("id")
            data = obj.get("data", "")

            if obj_type == "fact":
                store_fact(mem, data, user_id=user_id, agent_id=agent_id, app_id=app_id, run_id=run_id)
                continue

            if obj_type == "question":
                index_wait = float(os.getenv("MEM0_INDEX_WAIT_S", "0"))
                if index_wait > 0:
                    time.sleep(index_wait)
                messages = [{"role": "user", "content": data}]
                if args.debug:
                    reply, trace = agent.chat_with_trace(
                        messages=messages,
                        user_id=user_id,
                        agent_id=agent_id,
                        app_id=app_id,
                        run_id=run_id,
                    )
                else:
                    reply = agent.chat(
                        messages=messages,
                        user_id=user_id,
                        agent_id=agent_id,
                        app_id=app_id,
                        run_id=run_id,
                    )
                if args.debug:
                    print(
                        json.dumps(
                            {
                                "id": obj_id,
                                "debug_retrieval_query": trace.retrieval_query,
                                "debug_retrieved": [r.text for r in trace.memories],
                            },
                            ensure_ascii=False,
                        ),
                        file=sys.stderr,
                    )
                out_record = {"id": obj_id, "answer": reply}
                if args.debug:
                    out_record["debug_retrieval_query"] = trace.retrieval_query
                    out_record["debug_retrieved"] = [r.text for r in trace.memories]
                answers.append(out_record)
                continue

            # ignore unknown types

        # Save answers per patient under run/<task>/<run_id>/<user_id>.jsonl
        out_dir = ROOT / "run" / args.task / run_id
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{user_id}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for ans in answers:
                f.write(json.dumps(ans, ensure_ascii=False) + "\n")
        print(f"Results saved to: {out_path}")

        # Flush memory for this file after finishing
        if args.no_delete:
            if isinstance(mem, Mem0MemoryProvider):
                try:
                    index_wait = float(os.getenv("MEM0_INDEX_WAIT_S", "0"))
                    if index_wait > 0:
                        time.sleep(index_wait)
                    # Mem0 scopes memories by a single entity id; use run_id scope.
                    filters = {"AND": [{"run_id": run_id}]}
                    page = 1
                    all_ids = []
                    while True:
                        res = mem._client.get_all(filters=filters, page=page, page_size=200, version="v2")
                        items = res.get("results", []) if isinstance(res, dict) else res
                        if not items:
                            break
                        for it in items:
                            if isinstance(it, dict):
                                mid = it.get("memory_id") or it.get("id")
                                if mid:
                                    all_ids.append(mid)
                        page += 1
                    print(
                        json.dumps(
                            {
                                "item": item,
                                "user_id": user_id,
                                "run_id": run_id,
                                "memory_ids": all_ids,
                            },
                            ensure_ascii=False,
                        )
                    )
                except Exception as exc:
                    print(f"[warn] list memory ids failed: {exc}", file=sys.stderr)
            else:
                print(
                    json.dumps(
                        {
                            "item": item,
                            "user_id": user_id,
                            "run_id": run_id,
                            "memory_ids": [],
                            "note": "memory provider has no ids",
                        },
                        ensure_ascii=False,
                    )
                )
        else:
            maybe_flush(mem, user_id=user_id, agent_id=agent_id, app_id=app_id, run_id=run_id)


if __name__ == "__main__":
    main()
