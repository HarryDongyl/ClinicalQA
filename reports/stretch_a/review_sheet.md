# Stretch A evaluation review sheet (19 positives + 19 age-removed probes)

Check for each item: age/sex/creatinine extracted correctly; eGFR and stage plausible; the probe note reads naturally with no residual age cue; the question is sensible. Mark approve / fix / drop.

## sa_val_000 (source val_008)

- age **58**, sex **female**, creatinine **2.5 mg/dL** → eGFR **22**, stage **G4**
- age span: `HPI: 58-year-old female with a history of coronary arter`
- question: Estimate the patient's eGFR from the available serum creatinine, age and sex, and state the CKD stage.
- probe residue flags: none; extractor still finds age: False
- soft age mentions (no value; reviewer decides): none
- probe note (first 400 chars):

  > CC: Insomnia and early morning awakening for 2 months.  HPI: Female patient with a history of coronary artery disease presents with a 2-month history of difficulty maintaining sleep and early morning awakening, typically around 3–4 AM. She denies depressive symptoms, anxiety, caffeine use in the evening, or recent life stressors. She reports occasional mild fatigue and lower extremity swelling ove

- decision: [ ] approve  [ ] fix  [ ] drop

## sa_val_001 (source val_009)

- age **70**, sex **male**, creatinine **1.2 mg/dL** → eGFR **65**, stage **G2**
- age span: `HPI: Mr. J is a 70-year-old male presenting with a 6-week history o`
- question: Using the documented creatinine, age and sex, what is this patient's estimated GFR and corresponding CKD stage?
- probe residue flags: none; extractor still finds age: False
- soft age mentions (no value; reviewer decides): ['several months but attributed it to his age.']
- probe note (first 400 chars):

  > CC: Chronic low back pain radiating to the right leg for 6 weeks.  HPI: Mr. J is a male patient presenting with a 6-week history of progressive low back pain radiating down the right posterior thigh to the calf. He describes the pain as sharp and burning, rated 7/10, worsened with prolonged sitting and standing. He denies bowel or bladder incontinence, saddle anesthesia, or bilateral leg weakness.

- decision: [ ] approve  [ ] fix  [ ] drop

## sa_val_002 (source val_020)

- age **19**, sex **male**, creatinine **0.8 mg/dL** → eGFR **131**, stage **G1**
- age span: `**HPI:** A 19-year-old male with a known history of Parkinson`
- question: What is the patient's estimated glomerular filtration rate (CKD-EPI) based on their creatinine, age and sex?
- probe residue flags: none; extractor still finds age: False
- soft age mentions (no value; reviewer decides): none
- probe note (first 400 chars):

  > **ENCOUNTER NOTE**  **CC:** Insomnia and early morning awakening for 2 months.  **HPI:** A male patient with a known history of Parkinson disease presents with complaints of difficulty falling asleep and frequent early morning awakenings over the past 2 months. He reports sleeping only 3-4 hours per night, which has been affecting his daytime functioning and energy levels. He denies nightmares, re

- decision: [ ] approve  [ ] fix  [ ] drop

## sa_val_003 (source val_025)

- age **54**, sex **female**, creatinine **1.0 mg/dL** → eGFR **67**, stage **G2**
- age span: `HPI: 54-year-old female presents with her daughter who r`
- question: Calculate this patient's eGFR from the creatinine in the lab table and the patient's age and sex.
- probe residue flags: none; extractor still finds age: False
- soft age mentions (no value; reviewer decides): none
- probe note (first 400 chars):

  > CC: Confusion and falls over the past week.  HPI: Female patient presents with her daughter who reports increasing confusion, unsteady gait, and two falls at home over the past week. Patient has difficulty recalling recent events and appears more lethargic than baseline. She denies head trauma, seizures, chest pain, or shortness of breath. No recent medication changes. No fevers or sick contacts. 

- decision: [ ] approve  [ ] fix  [ ] drop

## sa_val_004 (source val_033)

- age **33**, sex **male**, creatinine **2.9 mg/dL** → eGFR **28**, stage **G4**
- age span: `HPI: 33-year-old male presenting for follow-up of progre`
- question: Estimate the patient's eGFR from the available serum creatinine, age and sex, and state the CKD stage.
- probe residue flags: none; extractor still finds age: False
- soft age mentions (no value; reviewer decides): none
- probe note (first 400 chars):

  > CLINICAL ENCOUNTER NOTE  CC: Bilateral hand tremor for 4 months.  HPI: Male patient presenting for follow-up of progressive bilateral hand tremor first noticed approximately 4 months ago. The tremor is present at rest and worsens with stress. He reports mild difficulty with fine motor tasks such as buttoning shirts. He denies falls, gait instability, or dysphagia. He also reports fatigue and inter

- decision: [ ] approve  [ ] fix  [ ] drop

## sa_val_005 (source val_036)

- age **71**, sex **male**, creatinine **1.0 mg/dL** → eGFR **80**, stage **G2**
- age span: `HPI: 71-year-old male presents with progressive bilatera`
- question: Using the documented creatinine, age and sex, what is this patient's estimated GFR and corresponding CKD stage?
- probe residue flags: none; extractor still finds age: False
- soft age mentions (no value; reviewer decides): none
- probe note (first 400 chars):

  > CC: Bilateral calf pain with ambulation.  HPI: Male patient presents with progressive bilateral calf pain occurring after walking approximately 200 meters, relieved within minutes of rest. Symptoms have worsened over the past 3 months. He denies rest pain, skin ulceration, or color changes in the lower extremities. He reports mild dyspnea at baseline consistent with known COPD. Today he feels slig

- decision: [ ] approve  [ ] fix  [ ] drop

## sa_val_006 (source val_044)

- age **78**, sex **female**, creatinine **0.7 mg/dL** → eGFR **88**, stage **G2**
- age span: `**HPI:** Mrs. J is a 78-year-old female presenting with a 2-month histor`
- question: What is the patient's estimated glomerular filtration rate (CKD-EPI) based on their creatinine, age and sex?
- probe residue flags: none; extractor still finds age: False
- soft age mentions (no value; reviewer decides): ['trigger avoidance. Refer for EGD given age and persistent symptoms to ru']
- probe note (first 400 chars):

  > **CLINICAL ENCOUNTER NOTE**  **CC:** Epigastric burning and acid reflux after meals for 2 months.  **HPI:** Mrs. J is a female patient presenting with a 2-month history of postprandial epigastric burning and acid reflux. She reports symptoms occur within 30 minutes of eating, worse with spicy or fatty foods, and occasionally awakens her at night. She denies dysphagia, odynophagia, hematemesis, mel

- decision: [ ] approve  [ ] fix  [ ] drop

## sa_val_007 (source val_059)

- age **57**, sex **female**, creatinine **4.0 mg/dL** → eGFR **12**, stage **G5**
- age span: `istory of Present Illness:** Ms. J is a 57-year-old female presenting with a persistent dry`
- question: Calculate this patient's eGFR from the creatinine in the lab table and the patient's age and sex.
- probe residue flags: none; extractor still finds age: False
- soft age mentions (no value; reviewer decides): none
- probe note (first 400 chars):

  > **Chief Complaint:** Dry cough and low-grade fever for 10 days.  **History of Present Illness:** Ms. J is a female patient presenting with a persistent dry, nonproductive cough and intermittent low-grade fevers over the past 10 days. She denies hemoptysis, chest pain, dyspnea, or night sweats. No recent travel or known sick contacts. She reports mild fatigue but no weight loss. She has been taking

- decision: [ ] approve  [ ] fix  [ ] drop

## sa_val_008 (source val_068)

- age **31**, sex **female**, creatinine **2.8 mg/dL** → eGFR **22**, stage **G4**
- age span: `HPI: 31-year-old female presents for evaluation of unint`
- question: Estimate the patient's eGFR from the available serum creatinine, age and sex, and state the CKD stage.
- probe residue flags: none; extractor still finds age: False
- soft age mentions (no value; reviewer decides): none
- probe note (first 400 chars):

  > CC: Unintentional weight loss of 8 kg over 3 months.  HPI: Female patient presents for evaluation of unintentional weight loss of approximately 8 kg over the past 3 months. She reports increased urinary frequency, fatigue, and occasional blurred vision. She denies fever, night sweats, changes in appetite, or gastrointestinal symptoms. She has not had recent changes to diet or exercise. She notes m

- decision: [ ] approve  [ ] fix  [ ] drop

## sa_val_009 (source val_070)

- age **71**, sex **male**, creatinine **2.1 mg/dL** → eGFR **33**, stage **G3b**
- age span: `HPI: Mr. J is a 71-year-old male presenting with worsening shortnes`
- question: Using the documented creatinine, age and sex, what is this patient's estimated GFR and corresponding CKD stage?
- probe residue flags: none; extractor still finds age: False
- soft age mentions (no value; reviewer decides): none
- probe note (first 400 chars):

  > CLINICAL ENCOUNTER NOTE  CC: Progressive dyspnea and orthopnea for 2 weeks.  HPI: Mr. J is a male patient presenting with worsening shortness of breath over the past 2 weeks. He reports dyspnea on exertion, now occurring with minimal activity such as walking across a room. He also endorses orthopnea requiring 3 pillows to sleep comfortably. He denies chest pain, palpitations, fever, or cough. No r

- decision: [ ] approve  [ ] fix  [ ] drop

## sa_val_010 (source val_086)

- age **61**, sex **male**, creatinine **0.7 mg/dL** → eGFR **105**, stage **G1**
- age span: `HPI: Mr. J is a 61-year-old male with a history of obesity, type 2`
- question: What is the patient's estimated glomerular filtration rate (CKD-EPI) based on their creatinine, age and sex?
- probe residue flags: none; extractor still finds age: False
- soft age mentions (no value; reviewer decides): none
- probe note (first 400 chars):

  > CC: Postprandial nausea and early satiety for 3 months.  HPI: Mr. J is a male patient with a history of obesity, type 2 diabetes mellitus, hyperlipidemia, pulmonary embolism, and HFpEF who presents with a 3-month history of postprandial nausea and early satiety. He reports feeling full after only a few bites of food, followed by nausea lasting 1-2 hours after meals. He denies vomiting, hematemesis

- decision: [ ] approve  [ ] fix  [ ] drop

## sa_val_011 (source val_110)

- age **30**, sex **female**, creatinine **3.6 mg/dL** → eGFR **17**, stage **G4**
- age span: `HPI: 30-year-old female presents with progressive bilate`
- question: Calculate this patient's eGFR from the creatinine in the lab table and the patient's age and sex.
- probe residue flags: none; extractor still finds age: False
- soft age mentions (no value; reviewer decides): none
- probe note (first 400 chars):

  > CC: Numbness and tingling in both feet for 2 months.  HPI: Female patient presents with progressive bilateral numbness and tingling in her feet over the past 2 months. She describes the sensation as starting in her toes and extending to the mid-foot. Symptoms are worse at night and occasionally interfere with sleep. She denies recent trauma, back pain, or weakness. No associated skin changes or ul

- decision: [ ] approve  [ ] fix  [ ] drop

## sa_val_012 (source val_130)

- age **74**, sex **male**, creatinine **1.1 mg/dL** → eGFR **70**, stage **G2**
- age span: `HPI: Mr. [REDACTED] is a 74-year-old male presenting with a 2-day history of`
- question: Estimate the patient's eGFR from the available serum creatinine, age and sex, and state the CKD stage.
- probe residue flags: none; extractor still finds age: False
- soft age mentions (no value; reviewer decides): none
- probe note (first 400 chars):

  > CC: Visual floaters and flashing lights in the right eye for 2 days.  HPI: Mr. [REDACTED] is a male patient presenting with a 2-day history of new-onset floaters and photopsia in the right eye. He describes multiple dark spots and intermittent flashing lights, predominantly in the temporal visual field. He denies vision loss, eye pain, or recent trauma. He has a history of migraine with aura but s

- decision: [ ] approve  [ ] fix  [ ] drop

## sa_val_013 (source val_141)

- age **81**, sex **female**, creatinine **0.8 mg/dL** → eGFR **74**, stage **G2**
- age span: `HPI: 81-year-old female presents with a 2-month history`
- question: Using the documented creatinine, age and sex, what is this patient's estimated GFR and corresponding CKD stage?
- probe residue flags: none; extractor still finds age: False
- soft age mentions (no value; reviewer decides): ['st, consider EGD referral given patient age and symptom duration.']
- probe note (first 400 chars):

  > CC: Epigastric burning and acid reflux after meals for 2 months.  HPI: Female patient presents with a 2-month history of progressive epigastric burning and acid reflux, occurring within 30-60 minutes after meals. She describes the discomfort as a burning sensation in the upper abdomen that radiates to the chest, often accompanied by sour taste in the mouth. Symptoms are worse with large meals and 

- decision: [ ] approve  [ ] fix  [ ] drop

## sa_val_014 (source val_192)

- age **62**, sex **female**, creatinine **3.7 mg/dL** → eGFR **13**, stage **G5**
- age span: `HPI: 62-year-old female presents with a 2-month history`
- question: What is the patient's estimated glomerular filtration rate (CKD-EPI) based on their creatinine, age and sex?
- probe residue flags: none; extractor still finds age: False
- soft age mentions (no value; reviewer decides): none
- probe note (first 400 chars):

  > CC: Decreased appetite and unintentional weight loss.  HPI: Female patient presents with a 2-month history of progressive appetite loss and approximately 5 kg of unintentional weight loss. She reports generalized fatigue and occasional nausea but denies vomiting, abdominal pain, hematemesis, melena, or hematochezia. She has noted increased urinary frequency and mild thirst. She denies fevers or ch

- decision: [ ] approve  [ ] fix  [ ] drop

## sa_val_015 (source val_194)

- age **83**, sex **female**, creatinine **1.2 mg/dL** → eGFR **45**, stage **G3a**
- age span: `HPI: 83-year-old female presents with a 6-week history o`
- question: Calculate this patient's eGFR from the creatinine in the lab table and the patient's age and sex.
- probe residue flags: none; extractor still finds age: False
- soft age mentions (no value; reviewer decides): none
- probe note (first 400 chars):

  > CC: Chronic low back pain radiating to the right leg for 6 weeks.  HPI: Female patient presents with a 6-week history of progressive low back pain radiating down the right posterior thigh to the knee. She describes the pain as constant, aching, rated 7/10, worsened by standing and walking. She denies bowel or bladder incontinence, saddle anesthesia, or bilateral leg weakness. She reports mild subj

- decision: [ ] approve  [ ] fix  [ ] drop

## sa_val_016 (source val_199)

- age **37**, sex **female**, creatinine **1.0 mg/dL** → eGFR **74**, stage **G2**
- age span: `**HPI:** 37-year-old female presenting for evaluation of rec`
- question: Estimate the patient's eGFR from the available serum creatinine, age and sex, and state the CKD stage.
- probe residue flags: none; extractor still finds age: False
- soft age mentions (no value; reviewer decides): none
- probe note (first 400 chars):

  > **CLINICAL ENCOUNTER NOTE**  **CC:** Recurring urinary tract infections over the past year.  **HPI:** Female patient presenting for evaluation of recurrent UTIs. Patient reports approximately 4-5 episodes over the past 12 months, each treated with short courses of antibiotics at urgent care facilities. She describes symptoms of dysuria, urinary frequency, and occasional suprapubic discomfort. She 

- decision: [ ] approve  [ ] fix  [ ] drop

## sa_val_017 (source val_223)

- age **85**, sex **male**, creatinine **0.9 mg/dL** → eGFR **84**, stage **G2**
- age span: `**HPI:** Mr. J is an 85-year-old male presenting with a 1-month history`
- question: Using the documented creatinine, age and sex, what is this patient's estimated GFR and corresponding CKD stage?
- probe residue flags: none; extractor still finds age: False
- soft age mentions (no value; reviewer decides): none
- probe note (first 400 chars):

  > **CLINICAL ENCOUNTER NOTE**  **CC:** Blurred vision and increased thirst for 1 month.  **HPI:** Mr. J is a male patient presenting with a 1-month history of progressive blurred vision and polydipsia. He reports drinking significantly more water than usual, approximately 3-4 liters daily, and notes intermittent blurring of vision, worse in the afternoons. He denies polyuria, weight loss, fever, hea

- decision: [ ] approve  [ ] fix  [ ] drop

## sa_val_018 (source val_233)

- age **43**, sex **female**, creatinine **1.2 mg/dL** → eGFR **58**, stage **G3a**
- age span: `HPI: 43-year-old female presents with progressive bilate`
- question: What is the patient's estimated glomerular filtration rate (CKD-EPI) based on their creatinine, age and sex?
- probe residue flags: none; extractor still finds age: False
- soft age mentions (no value; reviewer decides): none
- probe note (first 400 chars):

  > CLINICAL ENCOUNTER NOTE  CC: Worsening joint pain and morning stiffness for 4 months.  HPI: Female patient presents with progressive bilateral joint pain and morning stiffness over the past 4 months. She reports stiffness lasting approximately 90 minutes each morning, primarily affecting the metacarpophalangeal and proximal interphalangeal joints bilaterally. She denies fevers, rash, or recent ill

- decision: [ ] approve  [ ] fix  [ ] drop

