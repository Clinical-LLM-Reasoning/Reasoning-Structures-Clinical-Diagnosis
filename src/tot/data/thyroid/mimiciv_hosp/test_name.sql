SELECT DISTINCT di.itemid, di.label
FROM mimiciv_hosp.d_labitems di
WHERE LOWER(di.label) ~ 'thyroid|tsh|t3|t4|peroxidase'
ORDER BY di.label;
