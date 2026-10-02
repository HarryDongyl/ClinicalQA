"""Independent review of the Stretch A evaluation items (re-extraction, independent CKD-EPI 2021, probe diff, clinical scan).

    uv run python scripts/stretch_a_review.py    # prints one line per item; writes /tmp/sa_review/rows.json
"""
import os; os.makedirs("/tmp/sa_review", exist_ok=True)
import json, math, re, difflib
VAL = {json.loads(l)['id']: json.loads(l) for l in open('data/val.jsonl')}
POS = [json.loads(l) for l in open('data/stretch_a/val_egfr.jsonl')]
PRB = {json.loads(l)['source_id']: json.loads(l) for l in open('data/stretch_a/val_egfr_age_probes.jsonl')}

def egfr_indep(scr, age, female):
    # CKD-EPI 2021 (Inker et al., NEJM 2021), written independently of scripts/stretch_a_data.py
    k = 0.7 if female else 0.9
    a = -0.241 if female else -0.302
    v = 142 * (min(scr / k, 1.0) ** a) * (max(scr / k, 1.0) ** -1.200) * (0.9938 ** age) * (1.012 if female else 1.0)
    return v, math.floor(v + 0.5)

def stage(e):
    return 'G1' if e >= 90 else 'G2' if e >= 60 else 'G3a' if e >= 45 else 'G3b' if e >= 30 else 'G4' if e >= 15 else 'G5'

AGE = re.compile(r'\b(\d{1,3})[- ]year[- ]old\b', re.I)
SEXW = re.compile(r'\b(male|female|man|woman|gentleman|lady)\b', re.I)
PRON_F = re.compile(r'\b(she|her|hers|herself|Mrs\.|Ms\.)\b'); PRON_M = re.compile(r'\b(he|him|his|himself|Mr\.)\b')
CLIN = re.compile(r'acute kidney injury|\bAKI\b|acute or chronic kidney|acute renal|dialysis|hemodialysis|\bESRD\b|'
                  r'end-stage renal|pregnan|amputat|cachex|transplant|rhabdo', re.I)
rows = []
for p in POS:
    src = VAL[p['source_id']]; a = p['tool_calls'][0]['arguments']
    ages = sorted({int(m.group(1)) for m in AGE.finditer(src['note'])})
    sexw = {m.group(1).lower() for m in SEXW.finditer(src['note'])}
    nf, nm = len(PRON_F.findall(src['note'])), len(PRON_M.findall(src['note']))
    sex_note = 'female' if ({'female', 'woman', 'lady'} & sexw or nf > nm) else 'male' if ({'male', 'man', 'gentleman'} & sexw or nm > nf) else None
    cr_tab = [r[1] for r in src['table']['rows'] if r[0] == 'Creatinine']
    raw, e = egfr_indep(a['creatinine_mg_dl'], a['age'], a['sex'] == 'female')
    probe = PRB[p['source_id']]
    diff = [d for d in difflib.ndiff(src['note'].splitlines(), probe['note'].splitlines()) if d[:1] in '+-']
    removed_only_age = all(not AGE.search(d) for d in diff if d.startswith('+')) and any(AGE.search(d) for d in diff if d.startswith('-'))
    clin = sorted({m.group(0).lower() for m in CLIN.finditer(src['note'])})
    near_cut = min(abs(raw - c) for c in (90, 60, 45, 30, 15))
    issues = []
    if ages != [a['age']]: issues.append(f'age mentions {ages} vs arg {a["age"]}')
    if sex_note != a['sex']: issues.append(f'sex from note {sex_note} (words {sorted(sexw)}, she/her {nf}, he/his {nm}) vs arg {a["sex"]}')
    if cr_tab != [str(a['creatinine_mg_dl'])] and [float(x) for x in cr_tab] != [a['creatinine_mg_dl']]: issues.append(f'creatinine table {cr_tab} vs arg')
    if e != p['tool_calls'][0]['result']: issues.append(f'eGFR independent {e} ({raw:.2f}) vs stored {p["tool_calls"][0]["result"]}')
    if stage(e) not in p['answer'] or str(e) not in p['answer']: issues.append('answer missing value/category')
    if not removed_only_age: issues.append('probe diff is not a pure age removal')
    if AGE.search(probe['note']) or re.search(r'\b(twenties|thirties|forties|fifties|sixties|seventies|eighties|nineties)\b|\b\d0s\b', probe['note'], re.I): issues.append('numeric age residue')
    rows.append({'id': p['id'], 'src': p['source_id'], 'age': a['age'], 'sex': a['sex'], 'cr': a['creatinine_mg_dl'],
                 'egfr_raw': round(raw, 2), 'egfr': e, 'stage': stage(e), 'near_cutoff': round(near_cut, 2),
                 'clinical_flags': clin, 'soft_age': probe['stretch_a'].get('soft_age_mentions'), 'issues': issues,
                 'diff': [d[:2] + d[2:][:150] for d in diff]})
json.dump(rows, open('/tmp/sa_review/rows.json', 'w'), indent=1)
for r in rows:
    print(f"{r['id']} {r['src']} age={r['age']} {r['sex']:6s} cr={r['cr']:<4} eGFR={r['egfr_raw']:7.2f}->{r['egfr']:3d} {r['stage']:3s} "
          f"cut±{r['near_cutoff']:<5} clin={r['clinical_flags']} soft={len(r['soft_age'] or [])} ISSUES={r['issues']}")
