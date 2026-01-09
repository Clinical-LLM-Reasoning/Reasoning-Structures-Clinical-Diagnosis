-- Step 1: Identify subject_id + hadm_id with thyroid-related lab tests
WITH thyroid_labs AS (
    SELECT DISTINCT subject_id, hadm_id
    FROM mimiciv_hosp.labevents
    WHERE itemid IN (50992, 50993, 50994, 50995, 51001, 51721, 50896)  -- TSH, T3, T4, etc.
      AND hadm_id IS NOT NULL
),

-- Step 2: Discharge summaries
discharge_notes AS (
    SELECT subject_id, hadm_id, text AS discharge_text
    FROM mimiciv_note.discharge
),

-- Step 3: Radiology reports
radiology_notes AS (
    SELECT subject_id, hadm_id, text AS radiology_text
    FROM mimiciv_note.radiology
)

-- Step 4: Joint filtering, text must contain thyroid-related keywords
SELECT
  l.subject_id,
  l.hadm_id,
  d.discharge_text,
  r.radiology_text
FROM thyroid_labs l
JOIN discharge_notes d ON l.subject_id = d.subject_id AND l.hadm_id = d.hadm_id
JOIN radiology_notes r ON l.subject_id = r.subject_id AND l.hadm_id = r.hadm_id
WHERE
    LOWER(d.discharge_text) ~* 'thyroid|hypothyroid|hyperthyroid|tsh|t3|t4|levothyroxine|methimazole|goiter|graves'
 OR LOWER(r.radiology_text) ~* 'thyroid|nodule|goiter|t3|t4|hyperthyroid|hypothyroid'
LIMIT 100;
