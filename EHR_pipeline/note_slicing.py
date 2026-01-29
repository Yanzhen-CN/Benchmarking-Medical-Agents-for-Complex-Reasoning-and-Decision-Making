import re
import json
from typing import Dict, Optional, List, Tuple, Any

from config import BuildConfig

from util.logUtil import setup_logger
logger = setup_logger()

buildConfig = BuildConfig()
use_llm = getattr(buildConfig, "noteExtract", None) and buildConfig.noteExtract.USE_LLM
_llm = None
if use_llm:
    from util.llmUtil import LLMUtil
    _llm = LLMUtil()
    
# -----------------------------
# Helpers
# -----------------------------

def _clean(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    vv = v.strip()
    if not vv:
        return None
    if vv == "___":
        return None
    if re.fullmatch(r"_+", vv):
        return None
    return vv


def _force_schema(out: Dict[str, Any]) -> Dict[str, Any]:
    """Force output to match the target schema and clean placeholders."""
    adm = out.get("admission_info") or {}
    dis = out.get("discharge_info") or {}

    # admission_info must NOT include physical_exam
    if isinstance(adm, dict) and "physical_exam" in adm:
        adm.pop("physical_exam", None)

    admission_info = {
        "allergies": _clean(adm.get("allergies")),
        "attending": _clean(adm.get("attending")),
        "chief_complaint": _clean(adm.get("chief_complaint")),
        "history_of_present_illness": _clean(adm.get("history_of_present_illness")),
        "social_history": _clean(adm.get("social_history")),
        "family_history": _clean(adm.get("family_history")),
    }

    discharge_info = {
        "procedures": _clean(dis.get("procedures")),
        "physical_exam": _clean(dis.get("physical_exam")),
        "hospital_course": _clean(dis.get("hospital_course")),
        "discharge_disposition": _clean(dis.get("discharge_disposition")),
        "discharge_diagnosis": _clean(dis.get("discharge_diagnosis")),
        "discharge_condition": _clean(dis.get("discharge_condition")),
        "discharge_instructions": _clean(dis.get("discharge_instructions")),
        "followup_instructions": _clean(dis.get("followup_instructions")),
    }

    return {"admission_info": admission_info, "discharge_info": discharge_info}


# More robust discharge PE marker matching (covers common variants)
_DISCHARGE_PE_MARKERS = [
    r"Discharge\s*[:：]?\s*PHYSICAL\s+EXAMINATION\s*[:：]?",
    r"Discharge\s+Physical\s+Exam(?:ination)?\s*[:：]?",
    r"PHYSICAL\s+EXAMINATION\s+ON\s+DISCHARGE\s*[:：]?",
    r"DISCHARGE\s+EXAM(?:INATION)?\s*[:：]?",
]
_DISCHARGE_PE_PAT = re.compile(r"(?i)(" + "|".join(_DISCHARGE_PE_MARKERS) + r")", re.MULTILINE)


def _has_discharge_pe_marker(text: str) -> bool:
    return _DISCHARGE_PE_PAT.search(text) is not None


# -----------------------------
# Rule-based splitter (your original logic, unchanged except using _clean)
# -----------------------------

def split_note_to_adm_discharge_rule(text: str) -> Dict[str, Any]:
    """
    Rule-based split using heading slicing. Returns schema-compatible dict.
    """
    heading_order: List[str] = [
        "Allergies",
        "Attending",
        "Chief Complaint",
        "Major Surgical or Invasive Procedure",
        "History of Present Illness",
        "Social History",
        "Family History",
        "Physical Exam",
        "Brief Hospital Course",
        "Discharge Disposition",
        "Discharge Diagnosis",
        "Discharge Condition",
        "Discharge Instructions",
        "Followup Instructions",
    ]

    def _compile_heading_pat(h: str) -> re.Pattern:
        tokens = [re.escape(t) for t in h.split()]
        joined = r"\s+".join(tokens)
        return re.compile(rf"(?i)\b{joined}\b\s*[:：]", re.MULTILINE)

    pats: Dict[str, re.Pattern] = {h: _compile_heading_pat(h) for h in heading_order}

    # discharge physical exam marker inside Physical Exam block (legacy exact marker)
    discharge_pe_pat = re.compile(
        r"(?i)\bDischarge\s*[:：]\s*PHYSICAL\s+EXAMINATION\b\s*[:：]?",
        re.MULTILINE,
    )

    # Anchor at the first Allergies
    first_allergies = pats["Allergies"].search(text)
    if not first_allergies:
        return _force_schema({"admission_info": {}, "discharge_info": {}})

    matches: List[Tuple[str, int, int]] = [("Allergies", first_allergies.start(), first_allergies.end())]
    pos = first_allergies.end()

    for h in heading_order[1:]:
        m = pats[h].search(text, pos)
        if not m:
            continue
        matches.append((h, m.start(), m.end()))
        pos = m.end()

    if not matches:
        return _force_schema({"admission_info": {}, "discharge_info": {}})

    sections: Dict[str, Optional[str]] = {}
    for i, (h, _s, e) in enumerate(matches):
        next_s = matches[i + 1][1] if i + 1 < len(matches) else len(text)
        sections[h] = _clean(text[e:next_s])

    # discharge physical exam: only from discharge marker inside Physical Exam block
    pe_block = sections.get("Physical Exam")
    discharge_physical_exam: Optional[str] = None
    if pe_block:
        m = discharge_pe_pat.search(pe_block)
        if m:
            discharge_physical_exam = _clean(pe_block[m.end():])
        else:
            discharge_physical_exam = None

    admission_note = {
        "allergies": sections.get("Allergies"),
        "attending": sections.get("Attending"),
        "chief_complaint": sections.get("Chief Complaint"),
        "history_of_present_illness": sections.get("History of Present Illness"),
        "social_history": sections.get("Social History"),
        "family_history": sections.get("Family History"),
    }

    discharge_note = {
        "procedures": sections.get("Major Surgical or Invasive Procedure"),
        "physical_exam": discharge_physical_exam,
        "hospital_course": sections.get("Brief Hospital Course"),
        "discharge_disposition": sections.get("Discharge Disposition"),
        "discharge_diagnosis": sections.get("Discharge Diagnosis"),
        "discharge_condition": sections.get("Discharge Condition"),
        "discharge_instructions": sections.get("Discharge Instructions"),
        "followup_instructions": sections.get("Followup Instructions"),
    }

    return _force_schema({"admission_info": admission_note, "discharge_info": discharge_note})

def split_note_to_adm_discharge_llm(text: str) -> Dict[str, Any]:
    if not use_llm or _llm is None:
        raise RuntimeError("LLM not configured or disabled")
    system_prompt = """
You are an EHR note section extraction system. The user will provide you with a raw discharge note text.
Your task is to extract specific sections from the note and return them in a structured JSON format.
Return ONLY valid JSON (no markdown, no explanations).

Schema (must match exactly):
{
  "admission_info": {
    "allergies": string|null,
    "attending": string|null,
    "chief_complaint": string|null,
    "history_of_present_illness": string|null,
    "social_history": string|null,
    "family_history": string|null
  },
  "discharge_info": {
    "procedures": string|null,
    "physical_exam": string|null,
    "hospital_course": string|null,
    "discharge_disposition": string|null,
    "discharge_diagnosis": string|null,
    "discharge_condition": string|null,
    "discharge_instructions": string|null,
    "followup_instructions": string|null
  }
}

Rules:
- admission_info MUST NOT include physical_exam.
- Treat placeholders like "___" or strings of only "_" as missing -> null.
- Do NOT invent information. Extract verbatim spans from the raw note.
- discharge_info.physical_exam:
  Only extract discharge physical exam if the raw note contains a discharge physical exam marker such as:
  "Discharge: PHYSICAL EXAMINATION" / "Discharge Physical Exam" / "PHYSICAL EXAMINATION ON DISCHARGE" / "DISCHARGE EXAM".
  If there is no discharge physical exam marker, output null for physical_exam.
- Use the provided draft as a hint; correct it if needed.
"""

    llm_out = _llm.chat_json(
        system_prompt=system_prompt,
        user_text=text,
    )

    llm_out = _force_schema(llm_out)

    # Hard guard against hallucinating discharge PE
    if not _has_discharge_pe_marker(text):
        llm_out["discharge_info"]["physical_exam"] = None

    return llm_out

def split_note_to_adm_discharge(text: str) -> Dict[str, Any]:
    """
    Split the discharge note into admission_info and discharge_info.
    Uses LLM-based method if enabled; otherwise falls back to rule-based method.
    """
    if use_llm and _llm is not None:
        try:
            return split_note_to_adm_discharge_llm(text)
        except Exception as e:
            logger.warning(f"LLM-based note splitting failed, falling back to rule-based. Error: {e}")

    return split_note_to_adm_discharge_rule(text)

def _assert_contains(s: Optional[str], needle: str, msg: str) -> None:
    if s is None or needle not in s:
        raise AssertionError(msg)

def main():
    test_note = """Name: ___ Unit No: ___ Admission Date: ___ Discharge Date: ___ Date of Birth: ___ Sex: F Service: MEDICINE Allergies: No Known Allergies / Adverse Drug Reactions Attending: ___ Chief Complaint:Worsening ABD distension and pain Major Surgical or Invasive Procedure:Paracentesis History of Present Illness:___ HCV cirrhosis c/b ascites, hiv on ART, h/o IVDU, COPD, bioplar, PTSD, presented from OSH ED with worsening abd distension over past week. Pt reports self-discontinuing lasix and spirnolactone ___ weeks ago, because she feels like "they don't do anything" and that she "doesn't want to put more chemicals in her." She does not follow Na-restricted diets. In the past week, she notes that she has been having worsening abd distension and discomfort. She denies ___ edema, or SOB, or orthopnea. She denies f/c/n/v, d/c, dysuria. She had food poisoning a week ago from eating stale cake (n/v 20 min after food ingestion), which resolved the same day. She denies other recent illness or sick contacts. She notes that she has been noticing gum bleeding while brushing her teeth in recent weeks. she denies easy bruising, melena, BRBPR, hemetesis, hemoptysis, or hematuria. Because of her abd pain, she went to OSH ED and was transferred to ___ for further care. Per ED report, pt has brief period of confusion - she did not recall the ultrasound or bloodwork at osh. She denies recent drug use or alcohol use. She denies feeling confused, but reports that she is forgetful at times. In the ED, initial vitals were 98.4 70 106/63 16 97%RA Labs notable for ALT/AST/AP ___ ___: ___, Tbili1.6, WBC 5K, platelet 77, INR 1.6 Past Medical History:1. HCV Cirrhosis 2. No history of abnormal Pap smears. 3. She had calcification in her breast, which was removed previously and per patient not, it was benign. 4. For HIV disease, she is being followed by Dr. ___ Dr. ___. 5. COPD 6. Past history of smoking. 7. She also had a skin lesion, which was biopsied and showed skin cancer per patient report and is scheduled for a complete removal of the skin lesion in ___ of this year. 8. She also had another lesion in her forehead with purple discoloration. It was biopsied to exclude the possibility of ___'s sarcoma, the results is pending. 9. A 15 mm hypoechoic lesion on her ultrasound on ___ and is being monitored by an MRI. 10. History of dysplasia of anus in ___. 11. Bipolar affective disorder, currently manic, mild, and PTSD. 12. History of cocaine and heroin use. Social History:___Family History:She a total of five siblings, but she is not talking to most of them. She only has one brother that she is in touch with and lives in ___. She is not aware of any known GI or liver disease in her family. Her last alcohol consumption was one drink two months ago. No regular alcohol consumption. Last drug use ___ years ago. She quit smoking a couple of years ago. Physical Exam:VS: 98.1 107/61 78 18 97RA General: in NAD HEENT: CTAB, anicteric sclera, OP clear Neck: supple, no LAD CV: RRR,S1S2, no m/r/g Lungs: CTAb, prolonged expiratory phase, no w/r/r Abdomen: distended, mild diffuse tenderness, +flank dullness, cannot percuss liver/spleen edge ___ distension GU: no foley Ext: wwp, no c/e/e, + clubbing Neuro: AAO3, converse normally, able to recall 3 times after 5 minutes, CN II-XII intact Discharge:PHYSICAL EXAMINATION: VS: 98 105/70 95General: in NAD HEENT: anicteric sclera, OP clear Neck: supple, no LAD CV: RRR,S1S2, no m/r/g Lungs: CTAb, prolonged expiratory phase, no w/r/r Abdomen: distended but improved, TTP in RUQ, GU: no foley Ext: wwp, no c/e/e, + clubbing Neuro: AAO3, CN II-XII intact Pertinent Results:___ 10:25PM GLUCOSE-109* UREA N-25* CREAT-0.3* SODIUM-138 POTASSIUM-3.4 CHLORIDE-105 TOTAL CO2-27 ANION GAP-9___ 10:25PM estGFR-Using this___ 10:25PM ALT(SGPT)-100* AST(SGOT)-114* ALK PHOS-114* TOT BILI-1.6*___ 10:25PM LIPASE-77*___ 10:25PM ALBUMIN-3.3*___ 10:25PM WBC-5.0# RBC-4.29 HGB-14.3 HCT-42.6 MCV-99* MCH-33.3* MCHC-33.5 RDW-15.7*___ 10:25PM NEUTS-70.3* LYMPHS-16.5* MONOS-8.1 EOS-4.2* BASOS-0.8___ 10:25PM PLT COUNT-71*___ 10:25PM ___ PTT-30.9 ______ 10:25PM ___.CXR: No acute cardiopulmonary process. U/S: 1. Nodular appearance of the liver compatible with cirrhosis. Signs of portal hypertension including small amount of ascites and splenomegaly. 2. Cholelithiasis. 3. Patent portal veins with normal hepatopetal flow. Diagnostic para attempted in the ED, unsuccessful. On the floor, pt c/o abd distension and discomfort. Brief Hospital Course:___ HCV cirrhosis c/b ascites, hiv on ART, h/o IVDU, COPD, bioplar, PTSD, presented from OSH ED with worsening abd distension over past week and confusion. # Ascites - p/w worsening abd distension and discomfort for last week. likely ___ portal HTN given underlying liver disease, though no ascitic fluid available on night of admission. No signs of heart failure noted on exam. This was ___ to med non-compliance and lack of diet restriction. SBP negativediuretics: > Furosemide 40 mg PO DAILY > Spironolactone 50 mg PO DAILY, chosen over the usual 100mg dose d/t K+ of 4.5. CXR was wnl, UA negative, Urine culture blood culture negative. Pt was losing excess fluid appropriately with stable lytes on the above regimen. Pt was scheduled with current PCP for ___ check upon discharge. Pt was scheduled for new PCP with Dr. ___ at ___ and follow up in Liver clinic to schedule outpatient screening EGD and ___. Discharge Disposition:Home Discharge Diagnosis:Ascites from Portal HTN Discharge Condition:Mental Status: Clear and coherent.Level of Consciousness: Alert and interactive.Activity Status: Ambulatory - Independent. Discharge Instructions:Dear Ms. ___,It was a pleasure taking care of you! You came to us with stomach pain and worsening distension. While you were here we did a paracentesis to remove 1.5L of fluid from your belly. We also placed you on you 40 mg of Lasix and 50 mg of Aldactone to help you urinate the excess fluid still in your belly. As we discussed, everyone has a different dose of lasix required to make them urinate and it's likely that you weren't taking a high enough dose. Please take these medications daily to keep excess fluid off and eat a low salt diet. You will follow up with Dr. ___ in liver clinic and from there have your colonoscopy and EGD scheduled. Of course, we are always here if you need us. We wish you all the best!Your ___ Team. Followup Instructions:___"""

    print("Running note split test...\n")
    out = split_note_to_adm_discharge(test_note)

    assert "admission_info" in out and "discharge_info" in out
    adm = out["admission_info"]
    dis = out["discharge_info"]

    # admission must not have physical_exam
    assert "physical_exam" not in adm, "admission_info should not contain physical_exam"

    # basic sanity checks
    _assert_contains(adm.get("allergies"), "No Known Allergies", "Allergies not extracted correctly")
    _assert_contains(adm.get("chief_complaint"), "Worsening ABD distension and pain",
                     "Chief complaint not extracted correctly")
    _assert_contains(dis.get("discharge_diagnosis"), "Ascites from Portal HTN",
                     "Discharge diagnosis not extracted correctly")
    _assert_contains(dis.get("discharge_disposition"), "Home",
                     "Discharge disposition not extracted correctly")

    # discharge physical exam should prefer discharge exam block if marker exists
    assert dis.get("physical_exam") is not None, "discharge physical_exam should be extracted"
    _assert_contains(dis.get("physical_exam"), "VS: 98 105/70 95",
                     "Discharge physical exam not extracted correctly")

    print("Split succeeded.\n")
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print("\nAll tests passed.")


if __name__ == "__main__":
    main()