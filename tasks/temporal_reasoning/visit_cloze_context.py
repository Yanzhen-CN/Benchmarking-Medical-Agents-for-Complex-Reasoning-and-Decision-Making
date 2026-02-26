import json
import os
import glob
from tqdm import tqdm

def generate_cloze_context(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    files = glob.glob(os.path.join(input_dir, "P*.jsonl"))
    
    for fpath in tqdm(files, desc="Generating Cloze Context"):
        pid = os.path.basename(fpath).replace(".jsonl", "")
        with open(fpath, 'r', encoding='utf-8') as f:
            lines = [json.loads(line) for line in f]
            
        output_lines = []
        id_counter = 0
        timeline_stream = None
        
        for line in lines:
            if line['type'] == 'fact':
                timeline_stream = line['data'] 
            elif line['type'] == 'question' and timeline_stream:
                messages = [
                    {
                        "role": "system", 
                        "content": "You are analyzing a patient's clinical event stream. Some key events are missing, marked as [GAP]. Fill in the missing events based on the provided context. Think step by step to ensure accuracy: first understand the clinical timeline, identify patterns, then infer the most appropriate events to complete the sequence. Output only the completed event stream in the required format."
                    }
                ]
                
                # 逐条加入 ANCHOR 和 GAP
                for event in timeline_stream:
                    if event['type'] == 'ANCHOR':
                        msg = f"Event Time: {event.get('timestamp','N/A')} | Type: {event.get('event_type','-')} | Content: {event['content']}"
                        messages.append({"role": "user", "content": msg})
                    elif event['type'] == 'GAP':
                        messages.append({"role": "user", "content": f"[SYSTEM NOTIFICATION]: A clinical event is missing here: [GAP {event['gap_index']}]"})
                
                # 最后加入提问和格式要求
                messages.append({
                    "role": "user", 
                    "content": f"Task: {line['data']}\n\n[IMPORTANT]: Output strictly a JSON list of option IDs for each [GAP] in sequence. No text explanation. Example: [1, 0]"
                })
                
                output_lines.append({
                    "id": id_counter,
                    "messages": messages,
                    "ground_truth": line['ground_truth']
                })
                id_counter += 1
                
        if output_lines:
            with open(os.path.join(output_dir, f"{pid}.jsonl"), 'w', encoding='utf-8') as f_out:
                for entry in output_lines:
                    f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    generate_cloze_context("question_data/visit_cloze", "context_data/visit_cloze")