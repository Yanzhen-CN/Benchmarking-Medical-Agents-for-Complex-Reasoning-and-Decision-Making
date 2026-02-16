import json
import os
import glob
from pathlib import Path
from tqdm import tqdm

def generate_sorting_context(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    files = glob.glob(os.path.join(input_dir, "P*.jsonl"))

    for fpath in tqdm(files, desc="Generating Visit Sorting Context"):
        pid = os.path.basename(fpath).replace(".jsonl", "")
        with open(fpath, 'r', encoding='utf-8') as f:
            lines = [json.loads(line) for line in f]

        output_lines = []
        id_counter = 0
        current_options = None

        for line in lines:
            if line['type'] == 'fact':
                current_options = line['data']
            elif line['type'] == 'question' and current_options:
                snippets_text = "\n".join([f"[{k}]: {v}" for k, v in current_options.items()])

                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are a medical logic expert. Your task is to arrange the following clinical visit summaries "
                            "in chronological order based on clinical progression and logic. Think step by step to ensure accuracy. "
                            "Provide your reasoning internally, but output only the final JSON list of keys."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Here are the clinical visit summaries:\n{snippets_text}\n\n"
                            f"Task: {line['data']}\n\n"
                            "[IMPORTANT]: Output strictly a JSON list of integers representing the sequence of keys. "
                            "Do not include any explanation. Example: [0, 2, 1, 3]"
                        )
                    }
                ]

                output_lines.append({
                    "id": id_counter,
                    "messages": messages,
                    "ground_truth": line['ground_truth']
                })
                id_counter += 1

        if output_lines:
            out_path = os.path.join(output_dir, f"{pid}.jsonl")
            with open(out_path, 'w', encoding='utf-8') as f_out:
                for entry in output_lines:
                    f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    generate_sorting_context("question_data/visit_sorting", "context_data/visit_sorting")