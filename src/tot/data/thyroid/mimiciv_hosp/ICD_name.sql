SELECT 
    icd_code,
    long_title,
    icd_version
FROM mimiciv_hosp.d_icd_diagnoses
WHERE 
      (icd_version = 9  AND icd_code BETWEEN '240' AND '246')
   OR (icd_version = 10 AND icd_code ~ '^E0[0-7]')
ORDER BY icd_version, icd_code;
