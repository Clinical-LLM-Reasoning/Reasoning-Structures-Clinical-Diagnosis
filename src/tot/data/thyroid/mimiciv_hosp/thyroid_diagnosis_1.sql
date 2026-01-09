SELECT 
  d.subject_id,
  d.hadm_id,
  d.icd_code,
  d.icd_version,
  dd.long_title
FROM mimiciv_hosp.diagnoses_icd d
JOIN mimiciv_hosp.d_icd_diagnoses dd
  ON d.icd_code = dd.icd_code AND d.icd_version = dd.icd_version
WHERE 
  (d.icd_version = 9 AND d.icd_code BETWEEN '240' AND '246')
  OR
  (d.icd_version = 10 AND d.icd_code ~ '^E0[0-7]')
ORDER BY d.subject_id, d.hadm_id;
