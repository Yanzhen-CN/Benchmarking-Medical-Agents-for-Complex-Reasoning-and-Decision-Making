 ┌────────────────────────────────────────────────────────┐
 │                     1. Environment                     │
 │    (Patient Event Stream + Visibility Controller)      │
 │                                                        │
 │  ┌──────────────┐    ┌────────────────────────────┐    │
 │  │ Raw EHR Data │ →  │ Event Stream Constructor   │    │
 │  │ (multi-visit │    │ (sorted by timestamp)      │    │
 │  │ labs/notes/  │    └────────────────────────────┘    │
 │  │ meds/vitals) │                                      │
 │  └──────────────┘                                      │
 │                                                        │
 │  ┌─────────────────────────────┐   ┌────────────────┐  │
 │  │ History Visibility Manager  │ → │ Observation_t  │  │
 │  │  - full / window_k / none   │   │  (partial info │  │
 │  └─────────────────────────────┘   │   for LTM)     │  │
 │                                    └────────────────┘  │
 │                                                        │
 │  ┌──────────────────────────────────────────────────┐  │
 │  │ Probe Scheduler (memory test)                    │  │
 │  │  - fact recall                                   │  │
 │  │  - temporal recall                               │  │
 │  │  - trend recall                                  │  │
 │  │  - medication history recall                     │  │
 │  └──────────────────────────────────────────────────┘  │
 │              │                                         │
 └──────────────┼─────────────────────────────────────────┘
                │  Probe_t (question)
                ▼
        ┌────────────────────────────────────────────┐
        │                2. Agent                    │
        │        (LLM or memory-augmented agent)     │
        │--------------------------------------------│
        │ - Receives Observation_t                   │
        │ - Receives Probe_t (memory query)          │
        │ - NO need to call environment for recall   │
        │ - Just generates natural-language Answer_t │
        └────────────────────────────────────────────┘
                │
                ▼
 ┌────────────────────────────────────────────────────────┐
 │                   3. Memory Scorer                     │
 │--------------------------------------------------------│
 │ - Align Answer_t with ground-truth event stream        │
 │ - Exact / fuzzy matching                               │
 │ - Trend & temporal correctness check                   │
 │ - Produce score per probe + per episode                │
 └────────────────────────────────────────────────────────┘
