
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List


JsonObj = Dict[str, Any]
PARENT_DIR = Path(__file__).resolve().parent
JSON_DIR = PARENT_DIR / "bench_data" / "patients"
OUTPUT_PY_PATH = PARENT_DIR / "bench_data" / "patients" / "ehr_sequences.py"



# ***********************safe converting 

# not dict-> dict
def safe_dict(x: Any) -> JsonObj:
    return x if isinstance(x, dict) else {}

# not list-> list
def safe_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else []


# *******************normalize helper func
# replace \n 
_WS_RE = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    return _WS_RE.sub(" ", s).strip()

# output str
def format_primitive(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, str):
        return normalize_text(v)
    return normalize_text(str(v))


# **************** readable formatting (NOT JSON)
def format_value(v: Any, indent: int = 0) -> List[str]:
    
    pad = " " * indent
    lines: List[str] = []

    if isinstance(v, dict):
        for k, vv in v.items():  
            if isinstance(vv, (dict, list)):
                lines.append(f"{pad}{k}:")
                lines.extend(format_value(vv, indent + 2))
            else:
                lines.append(f"{pad}{k}: {format_primitive(vv)}")
        return lines

    if isinstance(v, list):
        for idx, item in enumerate(v):
            if isinstance(item, (dict, list)):
                lines.append(f"{pad}- [{idx}]")
                lines.extend(format_value(item, indent + 2))
            else:
                lines.append(f"{pad}- [{idx}] {format_primitive(item)}")
        return lines

    lines.append(pad + format_primitive(v))
    return lines


def make_tag(**fields: Any) -> str:
    """Lightweight tag string: KEY=VALUE | KEY=VALUE ..."""
    parts = []
    for k, v in fields.items():
        if v is None:
            continue
        parts.append(f"{k}={normalize_text(str(v))}")
    return " | ".join(parts)


# *****************build ONE patient's sequence (list[str]) 
def build_sequence_strings(doc: JsonObj, source_file: str) -> List[str]:
    seq: List[str] = []

    patient_info = safe_dict(doc.get("patient_info"))
    patient_id = patient_info.get("patient_id")
    subject_id = patient_info.get("subject_id")

    # 0) patient_info
    tag = make_tag(SECTION="PATIENT_INFO", patient_id=patient_id, subject_id=subject_id, source_file=source_file)
    body = "\n".join(format_value(patient_info, indent=0))
    seq.append(f"{tag}\n{body}".strip())

    # visits
    visits = safe_list(doc.get("visits"))
    for v in visits:
        v = safe_dict(v)
        visit_id = v.get("visit_id")
        hadm_id = v.get("hadm_id")

        # admission
        admission_info = safe_dict(v.get("admission_info"))
        adm_time = admission_info.get("admission_time")
        tag = make_tag(
            SECTION="ADMISSION_INFO",
            patient_id=patient_id,
            source_file=source_file,
            VISIT=visit_id,
            hadm_id=hadm_id,
            admission_time=adm_time,
        )
        body = "\n".join(format_value(admission_info, indent=0))
        seq.append(f"{tag}\n{body}".strip())

        # events
        event_stream = safe_list(v.get("event_stream"))
        for e_idx, e in enumerate(event_stream):
            e = safe_dict(e)
            e_type = e.get("type")
            ts = e.get("timestamp")
            event_id = e.get("event_id")

            tag = make_tag(
                SECTION="EVENT",
                patient_id=patient_id,
                source_file=source_file,
                VISIT=visit_id,
                hadm_id=hadm_id,
                event_index=e_idx,
                EVENT_TYPE=e_type,
                TIMESTAMP=ts,
                EVENT_ID=event_id,
            )
            body = "\n".join(format_value(e, indent=0))
            seq.append(f"{tag}\n{body}".strip())

        # discharge
        discharge_info = safe_dict(v.get("discharge_info"))
        dis_time = discharge_info.get("discharge_time")
        tag = make_tag(
            SECTION="DISCHARGE_INFO",
            patient_id=patient_id,
            source_file=source_file,
            VISIT=visit_id,
            hadm_id=hadm_id,
            discharge_time=dis_time,
        )
        body = "\n".join(format_value(discharge_info, indent=0))
        seq.append(f"{tag}\n{body}".strip())

    return seq


# *******************IO helpers 
def iter_input_files(input_arg: str) -> List[Path]:
    p = Path(input_arg)
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted(p.glob("*.json"))
    return sorted(Path().glob(input_arg))


def write_python_sequences(sequences: List[List[str]], source_files: List[str], patient_ids: List[Any], out_path: Path) -> None:
    
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def escape_triple(s: str) -> str:
        return s.replace('"""', r'\"\"\"')

    with out_path.open("w", encoding="utf-8") as f:
        f.write("sequences = [\n")
        for seq in sequences:
            f.write("    [\n")
            for item in seq:
                f.write('        """' + escape_triple(item) + '""",\n')
            f.write("    ],\n")
        f.write("]\n\n")

        # metadata lists (optional but useful)
        f.write("source_files = [\n")
        for sf in source_files:
            f.write("    " + repr(sf) + ",\n")
        f.write("]\n\n")

        f.write("patient_ids = [\n")
        for pid in patient_ids:
            f.write("    " + repr(pid) + ",\n")
        f.write("]\n")

# *************************** MAin
def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-patient sequences (list of lists) from multiple JSON files.")
    # parser.add_argument("-i", "--input", required=True, help="JSON file / directory / glob pattern (e.g., '*.json').")
    # parser.add_argument("-o", "--output", required=True, help="Output .py file path.")
    parser.add_argument("--encoding", default="utf-8", help="Input file encoding (default utf-8).")
    args = parser.parse_args()

    files = iter_input_files(JSON_DIR)
    if not files:
        raise SystemExit(f"No input files found for: {JSON_DIR}")

    out_path = Path(OUTPUT_PY_PATH)
    if out_path.suffix.lower() != ".py":
        raise SystemExit("Output must be a .py file.")

    sequences: List[List[str]] = []
    source_files: List[str] = []
    patient_ids: List[Any] = []

    for fp in files:
        try:
            with fp.open("r", encoding=args.encoding) as f:
                doc = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to read {fp}: {e}")
            continue

        if not isinstance(doc, dict):
            print(f"[WARN] Skip non-object root JSON: {fp}")
            continue

        patient_info = safe_dict(doc.get("patient_info"))
        pid = patient_info.get("patient_id")

        seq = build_sequence_strings(doc, source_file=str(fp))

        sequences.append(seq)
        source_files.append(str(fp))
        patient_ids.append(pid)

    write_python_sequences(sequences, source_files, patient_ids, out_path)
    print(f"[OK] Processed {len(sequences)} JSON files -> {out_path}")


if __name__ == "__main__":
    main()
