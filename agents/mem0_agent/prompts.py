DEFAULT_SYSTEM_PROMPT = (
    "You are a clinical reasoning assistant. Answer the user faithfully and "
    "succinctly using the provided conversation context and retrieved memories."
)

MEMORY_BLOCK_HEADER = "Relevant Memory"

OBSERVATION_EXTRACT_SYSTEM = (
    "Extract concise factual observations from the dialogue. "
    "Return ONLY a JSON array of short statements, no extra text."
)

DIALOG_STORE_TAG = "dialog"
OBSERVATION_STORE_TAG = "observation"
