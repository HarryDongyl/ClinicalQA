# P1 probe review packet (validation-derived synthetic probes)

Status: frozen after Codex AI grounding review on 2026-10-01. All 34 records unchanged. See docs/WAVE3_APPROVAL_REVIEW.md; this is not human clinical adjudication.

34 eligible of 40 grounded BMI sources; 6 excluded. Check that every weight/height/BMI mention is gone, nothing else changed meaning, and the note still reads naturally. Then `uv run python scripts/w3_prep.py approve p1 --reviewer <name>` freezes the set.

- excluded `val_017`: question states a measurement or BMI
- excluded `val_037`: question states a measurement or BMI
- excluded `val_139`: question states a measurement or BMI
- excluded `val_140`: question states a measurement or BMI
- excluded `val_189`: question states a measurement or BMI
- excluded `val_190`: question states a measurement or BMI

## p1_val_002 (mid-line edits: 0)

Question: What is this patient's BMI based on her recorded weight and height?

Removed table rows: none

```diff
@@ -14,2 +13,0 @@
-Weight: 93.7 kg. Height: 172.6 cm.
-
```

## p1_val_013 (mid-line edits: 1)

Question: What is this patient's BMI based on her recorded weight and height?

Removed table rows: none

```diff
@@ -10 +10 @@
-General: Alert, oriented, obese female in no acute distress. Weight 237.9 lb, Height 152.6 cm.
+General: Alert, oriented, obese female in no acute distress.
```

## p1_val_014 (mid-line edits: 1)

Question: What is this patient's BMI based on her recorded weight and height?

Removed table rows: none

```diff
@@ -9 +9 @@
-Vitals: BP 112/78 mmHg, HR 81 bpm, Temp 102.1°F, RR 18, SpO2 97%. Weight: 62.1 kg, Height: 187.2 cm.
+Vitals: BP 112/78 mmHg, HR 81 bpm, Temp 102.1°F, RR 18, SpO2 97%.
```

## p1_val_020 (mid-line edits: 1)

Question: What is this patient's body mass index (BMI) based on his recorded weight and height?

Removed table rows: none

```diff
@@ -15 +15 @@
-**Vitals:** BP 125/80 mmHg, HR 95 bpm, Temp 98.2 °F, RR 15 breaths/min, SpO2 96%. Weight 101.1 kg, Height 192.8 cm.
+**Vitals:** BP 125/80 mmHg, HR 95 bpm, Temp 98.2 °F, RR 15 breaths/min, SpO2 96%.
```

## p1_val_026 (mid-line edits: 1)

Question: What is the patient's BMI based on his recorded weight and height?

Removed table rows: none

```diff
@@ -9 +9 @@
-Exam: General: Alert, oriented, no acute distress. Weight 140.7 lb, Height 157.9 cm. HEENT: Right eye — pupils equal and reactive, visual acuity 20/25. Left eye — visual acuity hand motion only, relative afferent pupillary defect (RAPD) present. Fundoscopic exam of the left eye reveals pallor of the retina with a cherry-red spot at the macula, consistent with central retinal artery occlusion (CRAO). Box-carring noted in retinal arterioles. Right fundus unremarkable. CV: Regular rate and rhythm, no murmurs. Lungs: Decreased breath sounds bilaterally, no wheezing. Extremities: No edema.
+Exam: General: Alert, oriented, no acute distress. HEENT: Right eye — pupils equal and reactive, visual acuity 20/25. Left eye — visual acuity hand motion only, relative afferent pupillary defect (RAPD) present. Fundoscopic exam of the left eye reveals pallor of the retina with a cherry-red spot at the macula, consistent with central retinal artery occlusion (CRAO). Box-carring noted in retinal arterioles. Right fundus unremarkable. CV: Regular rate and rhythm, no murmurs. Lungs: Decreased breath sounds bilaterally, no wheezing. Extremities: No edema.
```

## p1_val_033 (mid-line edits: 1)

Question: What is this patient's BMI based on his recorded weight and height?

Removed table rows: none

```diff
@@ -13 +13 @@
-Vitals: BP 117/74 mmHg, HR 122 bpm, Temp 98.2°F, RR 15 breaths/min, SpO2 99%. Weight 145.7 lb, Height 72.6 in.
+Vitals: BP 117/74 mmHg, HR 122 bpm, Temp 98.2°F, RR 15 breaths/min, SpO2 99%.
```

## p1_val_038 (mid-line edits: 1)

Question: What is this patient's body mass index (BMI), and how does it factor into her clinical picture?

Removed table rows: none

```diff
@@ -9 +9 @@
-Exam: Vitals: BP 112/77, HR 86, Temp 101.6°F, RR 14, SpO2 92%. Weight 102.0 kg, height 150.7 cm. Patient is obese, ill-appearing. Scleral icterus noted. Abdomen soft, mild RUQ tenderness without rebound or guarding. Hyperactive bowel sounds. No hepatomegaly appreciated. Mild bilateral lower extremity edema.
+Exam: Vitals: BP 112/77, HR 86, Temp 101.6°F, RR 14, SpO2 92%. Patient is obese, ill-appearing. Scleral icterus noted. Abdomen soft, mild RUQ tenderness without rebound or guarding. Hyperactive bowel sounds. No hepatomegaly appreciated. Mild bilateral lower extremity edema.
```

## p1_val_052 (mid-line edits: 1)

Question: What is this patient's BMI based on her recorded weight and height?

Removed table rows: none

```diff
@@ -9 +9 @@
-Exam: Weight 101.0 kg, Height 63.1 in. BP 130/74 mmHg, HR 67 bpm, Temp 100.2 °F, RR 15, SpO2 92%. General: Mildly fatigued-appearing, no acute distress. Abdomen: Soft, mild diffuse tenderness in lower quadrants, no rebound or guarding, normoactive bowel sounds. Rectal exam: Trace bright red blood on glove. No perianal lesions.
+BP 130/74 mmHg, HR 67 bpm, Temp 100.2 °F, RR 15, SpO2 92%. General: Mildly fatigued-appearing, no acute distress. Abdomen: Soft, mild diffuse tenderness in lower quadrants, no rebound or guarding, normoactive bowel sounds. Rectal exam: Trace bright red blood on glove. No perianal lesions.
```

## p1_val_078 (mid-line edits: 0)

Question: What is this patient's BMI based on her recorded weight and height?

Removed table rows: none

```diff
@@ -13 +12,0 @@
-Weight: 54.1 kg, Height: 181.1 cm.
```

## p1_val_082 (mid-line edits: 1)

Question: What is this patient's body mass index (BMI) based on the documented weight and height?

Removed table rows: none

```diff
@@ -11 +11 @@
-Vitals: BP 127/72 mmHg, HR 68 bpm, Temp 101.9 °F, RR 18 breaths/min, SpO2 96% on room air. Weight 114.0 kg, Height 185.0 cm.
+Vitals: BP 127/72 mmHg, HR 68 bpm, Temp 101.9 °F, RR 18 breaths/min, SpO2 96% on room air.
```

## p1_val_086 (mid-line edits: 1)

Question: What is this patient's BMI based on his recorded weight and height?

Removed table rows: none

```diff
@@ -9 +9 @@
-Exam: Weight 221.3 lb, Height 157.5 cm. BP 122/78 mmHg, HR 97 bpm, Temp 98.2°F, RR 17, SpO2 97%. General: Obese male, no acute distress. Abdomen: Soft, non-tender, non-distended, normoactive bowel sounds, no organomegaly. Cardiac: Tachycardic, regular rhythm, no murmurs. Lungs: Clear bilaterally. Extremities: No edema.
+BP 122/78 mmHg, HR 97 bpm, Temp 98.2°F, RR 17, SpO2 97%. General: Obese male, no acute distress. Abdomen: Soft, non-tender, non-distended, normoactive bowel sounds, no organomegaly. Cardiac: Tachycardic, regular rhythm, no murmurs. Lungs: Clear bilaterally. Extremities: No edema.
```

## p1_val_094 (mid-line edits: 1)

Question: What is this patient's body mass index (BMI) based on her recorded weight and height?

Removed table rows: none

```diff
@@ -11 +11 @@
-General: Alert, oriented, obese female in no acute distress. Weight 92.8 kg, Height 170.2 cm.
+General: Alert, oriented, obese female in no acute distress.
```

## p1_val_097 (mid-line edits: 1)

Question: What is this patient's BMI based on his recorded weight and height?

Removed table rows: none

```diff
@@ -11 +11 @@
-Vitals: BP 130/72 mmHg, HR 73 bpm, Temp 98.7°F, RR 17 breaths/min, SpO2 98% on room air. Weight 50.5 kg, Height 71.7 in (182.2 cm).
+Vitals: BP 130/72 mmHg, HR 73 bpm, Temp 98.7°F, RR 17 breaths/min, SpO2 98% on room air.
```

## p1_val_104 (mid-line edits: 1)

Question: What is this patient's BMI based on her recorded weight and height?

Removed table rows: none

```diff
@@ -11 +11 @@
-Exam: VS: BP 147/92 mmHg, HR 88 bpm, Temp 98.1°F, RR 16, SpO2 97%. Weight 142.2 lb, Height 68.9 in. General: Alert, well-nourished female in no acute distress. Abdomen: Soft, mild tenderness in the left lower quadrant, no rebound or guarding, normoactive bowel sounds. Cardiac: Grade III/VI systolic ejection murmur at right upper sternal border, consistent with known AS. Rectal: Small amount of blood-tinged mucus on glove, no palpable masses.
+Exam: VS: BP 147/92 mmHg, HR 88 bpm, Temp 98.1°F, RR 16, SpO2 97%. General: Alert, well-nourished female in no acute distress. Abdomen: Soft, mild tenderness in the left lower quadrant, no rebound or guarding, normoactive bowel sounds. Cardiac: Grade III/VI systolic ejection murmur at right upper sternal border, consistent with known AS. Rectal: Small amount of blood-tinged mucus on glove, no palpable masses.
```

## p1_val_105 (mid-line edits: 1)

Question: What is this patient's BMI based on his recorded weight and height?

Removed table rows: none

```diff
@@ -11 +11 @@
-**Exam Findings:** Alert but disoriented to time and place. No focal neurologic deficits appreciated. Mild scleral icterus noted. Abdomen soft, non-tender, with mild hepatomegaly. No peripheral edema. Gait unsteady, wide-based. Weight 104.2 kg, Height 71.0 in.
+**Exam Findings:** Alert but disoriented to time and place. No focal neurologic deficits appreciated. Mild scleral icterus noted. Abdomen soft, non-tender, with mild hepatomegaly. No peripheral edema. Gait unsteady, wide-based.
```

## p1_val_112 (mid-line edits: 1)

Question: What is this patient's body mass index (BMI) based on his recorded weight and height, and how does it factor into his clinical picture?

Removed table rows: none

```diff
@@ -14 +14 @@
-General: Well-nourished male in mild discomfort. Weight 96.1 kg, height 190.8 cm.
+General: Well-nourished male in mild discomfort.
```

## p1_val_119 (mid-line edits: 1)

Question: What is this patient's BMI based on her recorded weight and height?

Removed table rows: none

```diff
@@ -9 +9 @@
-Vitals: BP 111/78 mmHg, HR 96 bpm, Temp 102.8°F, RR 16 breaths/min, SpO2 96%. Weight 78.2 kg, Height 70.7 in (179.7 cm).
+Vitals: BP 111/78 mmHg, HR 96 bpm, Temp 102.8°F, RR 16 breaths/min, SpO2 96%.
```

## p1_val_120 (mid-line edits: 1)

Question: What is this patient's body mass index (BMI), and how should it be interpreted in his clinical context?

Removed table rows: none

```diff
@@ -11 +11 @@
-**Vitals:** BP 116/85 mmHg, HR 89 bpm, Temp 102.9 °F, RR 17 breaths/min, SpO2 98% on room air. Weight 182.1 lb (82.6 kg), Height 165.9 cm.
+**Vitals:** BP 116/85 mmHg, HR 89 bpm, Temp 102.9 °F, RR 17 breaths/min, SpO2 98% on room air.
```

## p1_val_125 (mid-line edits: 0)

Question: What is this patient's BMI based on the recorded weight and height?

Removed table rows: none

```diff
@@ -13 +12,0 @@
-- Weight: 57.3 kg, Height: 175.2 cm
```

## p1_val_130 (mid-line edits: 0)

Question: What is this patient's BMI based on his recorded weight and height?

Removed table rows: none

```diff
@@ -11 +10,0 @@
-Weight: 61.5 kg, Height: 157.1 cm.
```

## p1_val_137 (mid-line edits: 1)

Question: What is this patient's BMI based on her recorded weight and height?

Removed table rows: none

```diff
@@ -11 +11 @@
-Vitals: BP 112/76 mmHg, HR 85 bpm, Temp 98.1°F, RR 27 breaths/min, SpO2 96%. Weight: 209.2 lb (94.9 kg), Height: 184.2 cm.
+Vitals: BP 112/76 mmHg, HR 85 bpm, Temp 98.1°F, RR 27 breaths/min, SpO2 96%.
```

## p1_val_150 (mid-line edits: 1)

Question: What is the patient's body mass index (BMI) based on her recorded weight and height?

Removed table rows: none

```diff
@@ -9 +9 @@
-Vitals: BP 165/103 mmHg, HR 84 bpm, Temp 97.7°F, RR 21 breaths/min, SpO2 96% on room air. Weight 164.0 lb, Height 183.1 cm.
+Vitals: BP 165/103 mmHg, HR 84 bpm, Temp 97.7°F, RR 21 breaths/min, SpO2 96% on room air.
```

## p1_val_155 (mid-line edits: 1)

Question: What is this patient's BMI based on her recorded weight and height?

Removed table rows: none

```diff
@@ -19 +19 @@
-Vitals: BP 126/79 mmHg, HR 60 bpm, Temp 98.3°F, RR 16, SpO2 96%. Weight 118.2 kg, Height 180.9 cm.
+Vitals: BP 126/79 mmHg, HR 60 bpm, Temp 98.3°F, RR 16, SpO2 96%.
```

## p1_val_158 (mid-line edits: 1)

Question: What is this patient's BMI based on her recorded weight and height?

Removed table rows: none

```diff
@@ -11 +11 @@
-Vitals: BP 111/77 mmHg, HR 135 bpm, Temp 101.0°F, RR 21 breaths/min, SpO2 97%. Weight 80.1 kg, Height 180.9 cm.
+Vitals: BP 111/77 mmHg, HR 135 bpm, Temp 101.0°F, RR 21 breaths/min, SpO2 97%.
```

## p1_val_159 (mid-line edits: 1)

Question: What is this patient's body mass index (BMI), and what clinical significance does it carry in the context of his presentation?

Removed table rows: none

```diff
@@ -11 +11 @@
-Vitals: BP 119/76 mmHg, HR 130 bpm, Temp 98.4°F, RR 18 breaths/min, SpO2 98% on room air. Weight 116.4 kg, Height 165.4 cm.
+Vitals: BP 119/76 mmHg, HR 130 bpm, Temp 98.4°F, RR 18 breaths/min, SpO2 98% on room air.
```

## p1_val_160 (mid-line edits: 0)

Question: What is this patient's BMI based on his documented weight and height?

Removed table rows: none

```diff
@@ -18 +17,0 @@
-Weight: 61.5 kg. Height: 64.4 in (163.7 cm).
```

## p1_val_164 (mid-line edits: 1)

Question: What is this patient's BMI based on the documented weight and height?

Removed table rows: none

```diff
@@ -9 +9 @@
-Vitals: BP 127/73 mmHg, HR 74 bpm, Temp 98.2°F, RR 15, SpO2 97%. Weight 118.9 kg, Height 194.0 cm.
+Vitals: BP 127/73 mmHg, HR 74 bpm, Temp 98.2°F, RR 15, SpO2 97%.
```

## p1_val_166 (mid-line edits: 1)

Question: What is this patient's body mass index (BMI) based on her recorded weight and height?

Removed table rows: none

```diff
@@ -9 +9 @@
-Exam: Weight 82.7 kg, Height 194.2 cm. BP 118/77, HR 82, Temp 100.8°F, RR 17, SpO2 92% on room air. Abdomen soft with moderate epigastric tenderness to palpation without guarding or rebound. No hepatomegaly. Mild bibasilar crackles on lung auscultation. Lower extremities without edema. See lab table below.
+BP 118/77, HR 82, Temp 100.8°F, RR 17, SpO2 92% on room air. Abdomen soft with moderate epigastric tenderness to palpation without guarding or rebound. No hepatomegaly. Mild bibasilar crackles on lung auscultation. Lower extremities without edema. See lab table below.
```

## p1_val_173 (mid-line edits: 1)

Question: What is this patient's BMI based on her recorded weight and height?

Removed table rows: none

```diff
@@ -11 +11 @@
-**Vitals:** BP 145/102 mmHg, HR 86 bpm, Temp 100.5°F, RR 17 breaths/min, SpO2 96%. Weight 117.1 kg, Height 159.1 cm.
+**Vitals:** BP 145/102 mmHg, HR 86 bpm, Temp 100.5°F, RR 17 breaths/min, SpO2 96%.
```

## p1_val_175 (mid-line edits: 1)

Question: What is the patient's BMI based on his recorded weight and height?

Removed table rows: none

```diff
@@ -11 +11 @@
-**Vitals:** BP 122/84 mmHg, HR 81 bpm, Temp 99.6 °F, RR 14 breaths/min, SpO2 94%. Weight 119.3 kg, Height 61.2 in (155.5 cm).
+**Vitals:** BP 122/84 mmHg, HR 81 bpm, Temp 99.6 °F, RR 14 breaths/min, SpO2 94%.
```

## p1_val_210 (mid-line edits: 1)

Question: What is this patient's BMI based on his recorded weight and height?

Removed table rows: none

```diff
@@ -13 +13 @@
-General: Obese male, no acute distress. Weight 232.8 lb, Height 65.0 in.
+General: Obese male, no acute distress.
```

## p1_val_213 (mid-line edits: 1)

Question: What is this patient's body mass index (BMI) based on her documented weight and height?

Removed table rows: none

```diff
@@ -19 +19 @@
-Vitals: BP 115/70 mmHg, HR 74 bpm, Temp 98.7°F, RR 18 breaths/min, SpO2 99%. Weight 253.5 lb, Height 156.9 cm.
+Vitals: BP 115/70 mmHg, HR 74 bpm, Temp 98.7°F, RR 18 breaths/min, SpO2 99%.
```

## p1_val_215 (mid-line edits: 1)

Question: What is this patient's body mass index (BMI)?

Removed table rows: none

```diff
@@ -12 +12 @@
-Vitals: BP 125/74 mmHg, HR 121 bpm, Temp 99.7°F, RR 14 breaths/min, SpO2 99% on room air. Weight 88.9 kg, Height 75.9 in (192.9 cm).
+Vitals: BP 125/74 mmHg, HR 121 bpm, Temp 99.7°F, RR 14 breaths/min, SpO2 99% on room air.
```

## p1_val_226 (mid-line edits: 1)

Question: What is this patient's BMI based on his recorded weight and height?

Removed table rows: none

```diff
@@ -9 +9 @@
-Vitals: BP 126/78 mmHg, HR 74 bpm, Temp 98.2 °F, RR 17 breaths/min, SpO2 93%. Weight 220.9 lb (100.2 kg), Height 185.2 cm.
+Vitals: BP 126/78 mmHg, HR 74 bpm, Temp 98.2 °F, RR 17 breaths/min, SpO2 93%.
```
