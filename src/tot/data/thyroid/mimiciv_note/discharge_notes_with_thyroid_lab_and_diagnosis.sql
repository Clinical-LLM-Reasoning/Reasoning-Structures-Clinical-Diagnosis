SELECT d.subject_id, d.hadm_id, d.storetime, d.text
FROM mimiciv_note.discharge d
WHERE d.subject_id IN (
  -- Patients who have undergone thyroid-related laboratory tests
  SELECT DISTINCT le.subject_id
  FROM mimiciv_hosp.labevents le
  WHERE le.itemid IN (50992, 50993, 50994, 50995, 51001, 51721, 50896)
)
AND d.subject_id IN (
  -- Patients diagnosed with thyroid disease
  SELECT DISTINCT di.subject_id
  FROM mimiciv_hosp.diagnoses_icd di
  WHERE (di.icd_version = 9 AND di.icd_code BETWEEN '240' AND '246')
     OR (di.icd_version = 10 AND di.icd_code LIKE 'E0[0-7]%')
);
