-- ================================================
-- STEP 1. Identify patients who have undergone thyroid tests
-- ================================================
WITH thyroid_patients AS (
    SELECT DISTINCT le.subject_id
    FROM mimiciv_hosp.labevents le
    WHERE le.itemid IN (50992, 50993, 50994, 50995, 51001, 51721, 50896)
),

-- ================================================
-- STEP 2. Identify patients who have NOT undergone thyroid tests
-- ================================================
non_thyroid_patients AS (
    SELECT p.subject_id
    FROM mimiciv_hosp.patients p
    LEFT JOIN thyroid_patients t ON p.subject_id = t.subject_id
    WHERE t.subject_id IS NULL
),

-- ================================================
-- STEP 3. Identify patients who have BOTH discharge summaries and radiology reports
-- ================================================
patients_with_both AS (
    SELECT d.subject_id
    FROM mimiciv_note.discharge d
    INNER JOIN mimiciv_note.radiology r
      ON d.subject_id = r.subject_id
    WHERE d.subject_id IN (SELECT subject_id FROM non_thyroid_patients)
    GROUP BY d.subject_id
),

-- ================================================
-- STEP 4. Randomly sample 2000 patients
-- ================================================
sampled_patients AS (
    SELECT subject_id
    FROM patients_with_both
    ORDER BY RANDOM()
    LIMIT 2000
),

-- ================================================
-- STEP 5. Extract discharge summaries and radiology reports for these patients
-- ================================================
discharge_notes AS (
    SELECT subject_id, hadm_id, text AS discharge_text
    FROM mimiciv_note.discharge
    WHERE subject_id IN (SELECT subject_id FROM sampled_patients)
),
radiology_notes AS (
    SELECT subject_id, hadm_id, text AS radiology_text
    FROM mimiciv_note.radiology
    WHERE subject_id IN (SELECT subject_id FROM sampled_patients)
)

-- ================================================
-- STEP 6. Export results (consistent with the thyroid cohort format)
-- ================================================
SELECT 
    COALESCE(d.subject_id, r.subject_id) AS subject_id,
    COALESCE(d.hadm_id, r.hadm_id) AS hadm_id,
    d.discharge_text,
    r.radiology_text
FROM discharge_notes d
FULL OUTER JOIN radiology_notes r
  ON d.subject_id = r.subject_id AND d.hadm_id = r.hadm_id
WHERE d.discharge_text IS NOT NULL AND r.radiology_text IS NOT NULL
ORDER BY RANDOM();
