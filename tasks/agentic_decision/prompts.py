TYPE_OF_MEMORY = {
    "discharge_notes": "discharge notes from previous visits",
    "event_stream": "chronologically ordered clinical events from previous visits",
    "event_context": "completed context of previous visits, including both discharge notes and event stream, organized by visit",
}

AGENT_ACTION_PROMPT="""
You are a doctor agent. You can perform these semantic actions when you are interacting with the environment: 
  ask_question, order_labs, order_microbiology, order_imaging, perform_procedure, medication, discharge. 
The environment returns only information that exists in this admission record; otherwise it returns not_available. Do not invent new findings.
Meanwhile, you may also asked to answer agentic-desision questions without interacting with the environment; in that case, you should answer based on the admission record and NOT perform any environment actions.
Use the following format for your messages:
  - If you are interacting with the environment, your message MUST be a JSON object with keys: {reason, action, args}. 
      Reason may explain workflow/clinical rationale but must not add new patient facts beyond the record.
  - If you are directly answering a question without environment interaction, your message MUST be a JSON object with keys: {reason, answer}.
    The answer should be a list containing only provided option from the question. You should return the raw text of the option(s) as the answer, not the label (e.g., "A", "B", ...). If multiple options are selected, include all of them in the list.For example, if the question is "What is the most likely diagnosis? 1. pneumonia 2. heart failure 3. COPD", and you think the answers are "pneumonia" and "heart failure", then your answer should be ["pneumonia", "heart failure"], not ["A", "B"]; if you think the answer is "pneumonia", then your answer should be ["pneumonia"], not ["1"].
    The reason should be a brief explanation based on the context and admission history.
"""

AGENT_QA_PROMPT = """
Now you are asked to answer agentic-desision questions without interacting with the environment; in that case, you should answer based on the admission record and NOT perform any environment actions.
Your message MUST be a JSON object with keys: {reason, answer}.
The answer should be a list containing only provided option from the question. You should return the raw text of the option(s) as the answer, not the label (e.g., "A", "B", ...). If multiple options are selected, include all of them in the list.For example, if the question is "What is the most likely diagnosis? 1. pneumonia 2. heart failure 3. COPD", and you think the answers are "pneumonia" and "heart failure", then your answer should be ["pneumonia", "heart failure"], not ["A", "B"]; if you think the answer is "pneumonia", then your answer should be ["pneumonia"], not ["1"].
The reason should be a brief explanation based on the context and admission history.
"""

def get_agent_prompt(type_of_memory: str) -> str:
    return AGENT_ACTION_PROMPT.format(type_of_memory=TYPE_OF_MEMORY.get(type_of_memory, "N/A"))
from typing import Any, Dict, List, Optional
def get_agent_qa_prompt() -> Dict[str, str]:
    return {"role": "system", "content": AGENT_QA_PROMPT}
  