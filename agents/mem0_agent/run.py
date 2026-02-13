from __future__ import annotations

from pathlib import Path
import os
import sys
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
)
from agents.mem0_agent.core import AgentConfig


def load_local_env() -> None:
    env_path = Path(__file__).parent / ".env"
    # Force local env to take precedence over repo/global .env
    dotenv.load_dotenv(env_path, override=True)


def build_agent() -> MemoryAugmentedChatAgent:
    llm = OpenAICompatibleLLMProvider()
    mem = Mem0MemoryProvider()
    obs = LLMObservationExtractor(llm)

    cfg = AgentConfig(
        max_recent_turns=int(os.getenv("AGENT_MAX_RECENT_TURNS", "6")),
        memory_top_k=int(os.getenv("MEMORY_TOP_K", "5")),
        store_dialog=os.getenv("AGENT_STORE_DIALOG", "1") == "1",
        store_observations=os.getenv("AGENT_STORE_OBS", "1") == "1",
        include_memory_in_prompt=os.getenv("AGENT_INCLUDE_MEMORY", "1") == "1",
        retrieval_policy=os.getenv("AGENT_RETRIEVAL_POLICY", "always"),
        query_rewrite=os.getenv("AGENT_QUERY_REWRITE", "1") == "1",
    )

    return MemoryAugmentedChatAgent(
        llm=llm,
        memory=mem,
        observation_extractor=obs,
        config=cfg,
    )


def main() -> None:
    load_local_env()

    agent = build_agent()
    messages = [
        {"role": "user", "content": "患者三天前开始发热并咳嗽。"}
    ]

    reply = agent.chat(
        messages=messages,
        user_id=os.getenv("PATIENT_ID", "P000001"),
        agent_id=os.getenv("AGENT_ID", "bench-agent"),
        app_id=os.getenv("AGENT_APP_ID", "medagentbench"),
        run_id=os.getenv("AGENT_RUN_ID", "task-001"),
    )
    print(reply)


if __name__ == "__main__":
    main()
