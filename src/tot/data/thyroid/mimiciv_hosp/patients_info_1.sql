SELECT DISTINCT p.subject_id, p.gender, p.anchor_age, p.anchor_year
FROM mimiciv_hosp.labevents le
JOIN mimiciv_hosp.d_labitems di ON le.itemid = di.itemid
JOIN mimiciv_hosp.patients p ON le.subject_id = p.subject_id
WHERE le.itemid IN (50992, 50993, 50994, 50995, 51001, 51721, 50896)
