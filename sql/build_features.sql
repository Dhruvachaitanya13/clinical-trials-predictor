-- Day 1: build one row per trial with structured features + the free-text summary.
-- Note: AACT occasionally changes status casing or column availability.
-- If a column errors, check the live data dictionary and adjust.

SELECT
    s.nct_id,
    s.phase,
    s.enrollment,
    s.number_of_arms,
    s.overall_status,
    EXTRACT(YEAR FROM s.start_date) AS start_year,
    (EXTRACT(YEAR FROM s.completion_date) - EXTRACT(YEAR FROM s.start_date)) * 12
        + (EXTRACT(MONTH FROM s.completion_date) - EXTRACT(MONTH FROM s.start_date)) AS duration_months,
    sp.agency_class            AS sponsor_class,
    d.allocation,
    d.intervention_model,
    d.masking,
    d.primary_purpose,
    (SELECT COUNT(*) FROM conditions c    WHERE c.nct_id = s.nct_id) AS n_conditions,
    (SELECT COUNT(*) FROM interventions i WHERE i.nct_id = s.nct_id) AS n_interventions,
    bs.description             AS brief_summary
FROM studies s
LEFT JOIN (
    SELECT DISTINCT ON (nct_id) nct_id, agency_class
    FROM sponsors
    WHERE lead_or_collaborator = 'lead'
) sp ON sp.nct_id = s.nct_id
LEFT JOIN designs d         ON d.nct_id = s.nct_id
LEFT JOIN brief_summaries bs ON bs.nct_id = s.nct_id
WHERE s.study_type = 'INTERVENTIONAL'
  AND s.phase IS NOT NULL
  AND s.phase <> 'NA'
  AND s.overall_status IN ('COMPLETED', 'TERMINATED', 'WITHDRAWN', 'SUSPENDED');
