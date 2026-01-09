SELECT 
  le.subject_id,
  le.hadm_id,
  le.charttime,
  le.storetime,
  le.itemid,
  di.label AS test_name,
  di.category,
  di.fluid,
  le.value,
  le.valuenum AS test_value,
  le.valueuom AS unit,
  le.ref_range_lower,
  le.ref_range_upper,
  le.flag,
  le.priority
FROM mimiciv_hosp.labevents le
JOIN mimiciv_hosp.d_labitems di
  ON le.itemid = di.itemid
WHERE le.itemid IN (50992, 50993, 50994, 50995, 51001, 51721, 50896)
ORDER BY le.subject_id, le.charttime;
