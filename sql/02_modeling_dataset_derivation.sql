-- 目的：按当前锁定边界生成严格主队列与宽松敏感性队列的 T0 前特征数据集，并执行质量核查。
-- 性质：仅创建当前 psql 会话可见的 TEMP TABLE；不创建持久表、不训练模型。
-- 结局：T0 后至 min(ICU 入科时间+7 d, 住院出院, 死亡) 前任一血钠 Na >=151 mmol/L 的已记录事件。
-- 重要：T0 后检测次数、事件时间和未来观察时长不导出为模型预测变量；无复查血钠者不进入模型数据集。

\pset pager off
SET enable_nestloop = off;
SET work_mem = '256MB';

CREATE TEMP TABLE hypernatremia_feature_base_v01 AS
WITH ranked_icu AS (
    SELECT i.subject_id, i.hadm_id, i.stay_id, i.intime, i.outtime, a.dischtime, a.deathtime,
           p.gender, p.anchor_year_group,
           p.anchor_age + EXTRACT(YEAR FROM a.admittime) - p.anchor_year AS age,
           ROW_NUMBER() OVER (PARTITION BY i.subject_id ORDER BY i.intime, i.stay_id) AS icu_rank
    FROM mimiciv_icu.icustays AS i
    JOIN mimiciv_hosp.admissions AS a ON i.subject_id = a.subject_id AND i.hadm_id = a.hadm_id
    JOIN mimiciv_hosp.patients AS p ON i.subject_id = p.subject_id
), t0_risk_set AS (
    SELECT *, intime + INTERVAL '24 hour' AS t0_time,
           LEAST(intime + INTERVAL '7 day', dischtime, COALESCE(deathtime, dischtime)) AS observation_end
    FROM ranked_icu
    WHERE icu_rank = 1 AND age >= 18
      AND outtime > intime + INTERVAL '24 hour'
      AND dischtime > intime + INTERVAL '24 hour'
      AND (deathtime IS NULL OR deathtime > intime + INTERVAL '24 hour')
), sodium_window AS MATERIALIZED (
    SELECT r.stay_id,
           COUNT(c.sodium) FILTER (WHERE c.charttime < r.t0_time) AS sodium_n_first24,
           MIN(c.sodium) FILTER (WHERE c.charttime < r.t0_time) AS sodium_min_first24,
           MAX(c.sodium) FILTER (WHERE c.charttime < r.t0_time) AS sodium_max_first24,
           COUNT(c.sodium) FILTER (WHERE c.charttime >= r.t0_time) AS sodium_n_followup,
           MIN(c.charttime) FILTER (WHERE c.charttime >= r.t0_time AND c.sodium >= 151) AS first_na_ge_151_time
    FROM t0_risk_set AS r
    LEFT JOIN mimiciv_derived.chemistry AS c
      ON c.hadm_id = r.hadm_id AND c.charttime >= r.intime
     AND c.charttime < r.observation_end AND c.sodium IS NOT NULL
    GROUP BY r.stay_id
), normal_sodium_risk_set AS (
    SELECT r.*, s.sodium_n_first24, s.sodium_n_followup, s.first_na_ge_151_time
    FROM t0_risk_set AS r JOIN sodium_window AS s USING (stay_id)
    WHERE s.sodium_n_first24 >= 1 AND s.sodium_min_first24 >= 135 AND s.sodium_max_first24 <= 145
), treatment_flags AS MATERIALIZED (
    SELECT r.stay_id,
           BOOL_OR(i.itemid IN (225161, 228341)) AS hypertonic_saline_pre_t0,
           BOOL_OR(i.itemid = 227531) AS mannitol_pre_t0
    FROM normal_sodium_risk_set AS r
    LEFT JOIN mimiciv_icu.inputevents AS i
      ON i.stay_id = r.stay_id AND i.starttime < r.t0_time
     AND COALESCE(i.endtime, i.starttime) > r.intime
     AND i.itemid IN (225161, 228341, 227531)
    GROUP BY r.stay_id
), model_cohort AS (
    SELECT r.*, COALESCE(f.mannitol_pre_t0, false) AS mannitol_pre_t0,
           (r.sodium_n_first24 >= 2) AS is_strict_primary,
           (r.first_na_ge_151_time IS NOT NULL) AS outcome_na_ge_151
    FROM normal_sodium_risk_set AS r
    LEFT JOIN treatment_flags AS f USING (stay_id)
    WHERE NOT COALESCE(f.hypertonic_saline_pre_t0, false)
      AND r.sodium_n_followup > 0
), chemistry_features AS (
    SELECT p.stay_id, MAX(c.charttime) AS chemistry_last_charttime,
           (ARRAY_AGG(c.sodium ORDER BY c.charttime) FILTER (WHERE c.sodium IS NOT NULL))[1] AS sodium_first,
           (ARRAY_AGG(c.sodium ORDER BY c.charttime DESC) FILTER (WHERE c.sodium IS NOT NULL))[1] AS sodium_last,
           MIN(c.sodium) AS sodium_min, MAX(c.sodium) AS sodium_max, COUNT(c.sodium) AS sodium_n,
           (ARRAY_AGG(c.creatinine ORDER BY c.charttime DESC) FILTER (WHERE c.creatinine IS NOT NULL))[1] AS creatinine_last,
           (ARRAY_AGG(c.bun ORDER BY c.charttime DESC) FILTER (WHERE c.bun IS NOT NULL))[1] AS bun_last,
           (ARRAY_AGG(c.chloride ORDER BY c.charttime DESC) FILTER (WHERE c.chloride IS NOT NULL))[1] AS chloride_last,
           (ARRAY_AGG(c.potassium ORDER BY c.charttime DESC) FILTER (WHERE c.potassium IS NOT NULL))[1] AS potassium_last,
           (ARRAY_AGG(c.bicarbonate ORDER BY c.charttime DESC) FILTER (WHERE c.bicarbonate IS NOT NULL))[1] AS bicarbonate_last,
           (ARRAY_AGG(c.glucose ORDER BY c.charttime DESC) FILTER (WHERE c.glucose IS NOT NULL))[1] AS glucose_last
    FROM model_cohort AS p
    LEFT JOIN mimiciv_derived.chemistry AS c
      ON c.hadm_id = p.hadm_id AND c.charttime >= p.intime AND c.charttime < p.t0_time
    GROUP BY p.stay_id
), vital_features AS (
    SELECT p.stay_id, MAX(v.charttime) AS vital_last_charttime,
           (ARRAY_AGG(v.heart_rate ORDER BY v.charttime DESC) FILTER (WHERE v.heart_rate IS NOT NULL))[1] AS heart_rate_last,
           (ARRAY_AGG(v.mbp ORDER BY v.charttime DESC) FILTER (WHERE v.mbp IS NOT NULL))[1] AS mbp_last,
           (ARRAY_AGG(v.resp_rate ORDER BY v.charttime DESC) FILTER (WHERE v.resp_rate IS NOT NULL))[1] AS resp_rate_last,
           (ARRAY_AGG(v.spo2 ORDER BY v.charttime DESC) FILTER (WHERE v.spo2 IS NOT NULL))[1] AS spo2_last,
           (ARRAY_AGG(v.temperature ORDER BY v.charttime DESC) FILTER (WHERE v.temperature IS NOT NULL))[1] AS temperature_last
    FROM model_cohort AS p
    LEFT JOIN mimiciv_derived.vitalsign AS v
      ON v.stay_id = p.stay_id AND v.charttime >= p.intime AND v.charttime < p.t0_time
    GROUP BY p.stay_id
), urine_features AS (
    SELECT p.stay_id, MAX(u.charttime) AS urine_last_charttime, SUM(u.urineoutput) AS urineoutput_total
    FROM model_cohort AS p
    LEFT JOIN mimiciv_derived.urine_output AS u
      ON u.stay_id = p.stay_id AND u.charttime >= p.intime AND u.charttime < p.t0_time
    GROUP BY p.stay_id
), urine_rate_features AS (
    SELECT p.stay_id, MAX(u.charttime) AS uo_rate_last_charttime,
           (ARRAY_AGG(u.uo_mlkghr_6hr ORDER BY u.charttime DESC) FILTER (WHERE u.uo_mlkghr_6hr IS NOT NULL))[1] AS uo_mlkghr_6hr_last,
           (ARRAY_AGG(u.uo_mlkghr_12hr ORDER BY u.charttime DESC) FILTER (WHERE u.uo_mlkghr_12hr IS NOT NULL))[1] AS uo_mlkghr_12hr_last
    FROM model_cohort AS p
    LEFT JOIN mimiciv_derived.urine_output_rate AS u
      ON u.stay_id = p.stay_id AND u.charttime >= p.intime AND u.charttime < p.t0_time
    GROUP BY p.stay_id
), ventilation_features AS (
    SELECT p.stay_id, MAX(v.starttime) AS ventilation_latest_starttime,
           COALESCE(BOOL_OR(v.ventilation_status = 'InvasiveVent'), false) AS invasive_vent_pre_t0,
           COALESCE(BOOL_OR(v.ventilation_status = 'NonInvasiveVent'), false) AS noninvasive_vent_pre_t0,
           COALESCE(BOOL_OR(v.ventilation_status = 'HFNC'), false) AS hfnc_pre_t0,
           COALESCE(BOOL_OR(v.ventilation_status = 'SupplementalOxygen'), false) AS supplemental_oxygen_pre_t0,
           COALESCE(BOOL_OR(v.ventilation_status = 'Tracheostomy'), false) AS tracheostomy_pre_t0,
           COUNT(v.stay_id) = 0 AS ventilation_no_record_pre_t0
    FROM model_cohort AS p
    LEFT JOIN mimiciv_derived.ventilation AS v
      ON v.stay_id = p.stay_id AND v.starttime < p.t0_time
     AND COALESCE(v.endtime, v.starttime) > p.intime
    GROUP BY p.stay_id
), treatment_features AS (
    SELECT p.stay_id,
           COALESCE(BOOL_OR(i.itemid IN (221794, 228340, 229639)), false) AS loop_diuretic_pre_t0,
           COALESCE(BOOL_OR(i.itemid IN (221906, 221289, 229617, 221662, 221653, 221749, 229630, 229632, 222315)), false) AS vasoactive_pre_t0,
           COALESCE(BOOL_OR(i.itemid IN (225158, 225828) AND i.ordercategoryname IN ('02-Fluids (Crystalloids)', '03-IV Fluid Bolus')), false) AS iv_isotonic_crystalloid_pre_t0,
           COALESCE(BOOL_OR(i.itemid IN (220949, 225823, 225825, 225827, 225159) AND i.ordercategoryname IN ('02-Fluids (Crystalloids)', '03-IV Fluid Bolus')), false) AS dextrose_or_hypotonic_fluid_pre_t0,
           MAX(i.starttime) FILTER (WHERE i.itemid IN (221794, 228340, 229639, 221906, 221289, 229617, 221662, 221653, 221749, 229630, 229632, 222315, 225158, 225828, 220949, 225823, 225825, 225827, 225159)) AS treatment_latest_starttime
    FROM model_cohort AS p
    LEFT JOIN mimiciv_icu.inputevents AS i
      ON i.stay_id = p.stay_id AND i.starttime < p.t0_time AND COALESCE(i.endtime, i.starttime) > p.intime
    GROUP BY p.stay_id
), rrt_features AS (
    SELECT p.stay_id,
           COALESCE(BOOL_OR(e.itemid IN (225802, 225441, 225805)), false) AS rrt_pre_t0,
           MAX(e.starttime) FILTER (WHERE e.itemid IN (225802, 225441, 225805)) AS rrt_latest_starttime
    FROM model_cohort AS p
    LEFT JOIN mimiciv_icu.procedureevents AS e
      ON e.stay_id = p.stay_id AND e.starttime < p.t0_time AND COALESCE(e.endtime, e.starttime) > p.intime
    GROUP BY p.stay_id
)
SELECT p.subject_id, p.hadm_id, p.stay_id, p.anchor_year_group,
       CASE WHEN substring(p.anchor_year_group FROM 1 FOR 4)::int <= 2016 THEN 'development_2008_2016' ELSE 'temporal_test_2017_2022' END AS temporal_split,
       p.is_strict_primary, p.outcome_na_ge_151, p.mannitol_pre_t0,
       p.intime, p.t0_time, p.observation_end, p.first_na_ge_151_time, p.sodium_n_followup,
       p.age, p.gender,
       c.chemistry_last_charttime, c.sodium_first, c.sodium_last, c.sodium_min, c.sodium_max, c.sodium_last - c.sodium_first AS sodium_delta, c.sodium_n,
       c.creatinine_last, c.bun_last, c.chloride_last, c.potassium_last, c.bicarbonate_last, c.glucose_last,
       v.vital_last_charttime, v.heart_rate_last, v.mbp_last, v.resp_rate_last, v.spo2_last, v.temperature_last,
       u.urine_last_charttime, u.urineoutput_total,
       ur.uo_rate_last_charttime, ur.uo_mlkghr_6hr_last, ur.uo_mlkghr_12hr_last,
       ve.ventilation_latest_starttime, ve.invasive_vent_pre_t0, ve.noninvasive_vent_pre_t0, ve.hfnc_pre_t0, ve.supplemental_oxygen_pre_t0, ve.tracheostomy_pre_t0, ve.ventilation_no_record_pre_t0,
       t.loop_diuretic_pre_t0, t.vasoactive_pre_t0, t.iv_isotonic_crystalloid_pre_t0, t.dextrose_or_hypotonic_fluid_pre_t0, t.treatment_latest_starttime,
       r.rrt_pre_t0, r.rrt_latest_starttime
FROM model_cohort AS p
LEFT JOIN chemistry_features AS c USING (stay_id)
LEFT JOIN vital_features AS v USING (stay_id)
LEFT JOIN urine_features AS u USING (stay_id)
LEFT JOIN urine_rate_features AS ur USING (stay_id)
LEFT JOIN ventilation_features AS ve USING (stay_id)
LEFT JOIN treatment_features AS t USING (stay_id)
LEFT JOIN rrt_features AS r USING (stay_id);

-- A. 队列、结局与时间切分核查。
SELECT CASE WHEN is_strict_primary THEN 'strict_primary' ELSE 'loose_extra_one_measurement' END AS cohort_component,
       temporal_split, COUNT(*) AS stays, SUM(outcome_na_ge_151::int) AS events,
       ROUND(100.0 * AVG(outcome_na_ge_151::int), 2) AS observed_event_pct
FROM hypernatremia_feature_base_v01
GROUP BY cohort_component, temporal_split
ORDER BY cohort_component, temporal_split;

-- B. 一行一个 stay 与时间截断核查。所有计数预期为 0。
SELECT COUNT(*) AS rows_total,
       COUNT(DISTINCT stay_id) AS distinct_stays,
       COUNT(*) - COUNT(DISTINCT stay_id) AS duplicate_stay_rows,
       COUNT(*) FILTER (WHERE chemistry_last_charttime >= t0_time) AS chemistry_at_or_after_t0,
       COUNT(*) FILTER (WHERE vital_last_charttime >= t0_time) AS vital_at_or_after_t0,
       COUNT(*) FILTER (WHERE urine_last_charttime >= t0_time) AS urine_at_or_after_t0,
       COUNT(*) FILTER (WHERE uo_rate_last_charttime >= t0_time) AS urine_rate_at_or_after_t0,
       COUNT(*) FILTER (WHERE ventilation_latest_starttime >= t0_time) AS ventilation_start_at_or_after_t0,
       COUNT(*) FILTER (WHERE treatment_latest_starttime >= t0_time) AS treatment_start_at_or_after_t0,
       COUNT(*) FILTER (WHERE rrt_latest_starttime >= t0_time) AS rrt_start_at_or_after_t0
FROM hypernatremia_feature_base_v01;

-- C. 连续特征缺失审计。二元治疗/通气变量在提取阶段已编码为 false 或独立无记录状态。
SELECT COUNT(*) FILTER (WHERE creatinine_last IS NULL) AS missing_creatinine_last,
       COUNT(*) FILTER (WHERE bun_last IS NULL) AS missing_bun_last,
       COUNT(*) FILTER (WHERE chloride_last IS NULL) AS missing_chloride_last,
       COUNT(*) FILTER (WHERE potassium_last IS NULL) AS missing_potassium_last,
       COUNT(*) FILTER (WHERE bicarbonate_last IS NULL) AS missing_bicarbonate_last,
       COUNT(*) FILTER (WHERE glucose_last IS NULL) AS missing_glucose_last,
       COUNT(*) FILTER (WHERE heart_rate_last IS NULL) AS missing_heart_rate_last,
       COUNT(*) FILTER (WHERE mbp_last IS NULL) AS missing_mbp_last,
       COUNT(*) FILTER (WHERE resp_rate_last IS NULL) AS missing_resp_rate_last,
       COUNT(*) FILTER (WHERE spo2_last IS NULL) AS missing_spo2_last,
       COUNT(*) FILTER (WHERE temperature_last IS NULL) AS missing_temperature_last,
       COUNT(*) FILTER (WHERE urineoutput_total IS NULL) AS missing_urineoutput_total,
       COUNT(*) FILTER (WHERE uo_mlkghr_6hr_last IS NULL) AS missing_uo_6hr_last,
       COUNT(*) FILTER (WHERE uo_mlkghr_12hr_last IS NULL) AS missing_uo_12hr_last
FROM hypernatremia_feature_base_v01;

-- D. 导出：CSV 只含 ID、结局、时间切分和 T0 前候选特征；不含 T0 后检测次数、事件时间或未来观察时长。
\copy (SELECT subject_id, hadm_id, stay_id, anchor_year_group, temporal_split, outcome_na_ge_151, mannitol_pre_t0, age, gender, sodium_first, sodium_last, sodium_min, sodium_max, sodium_delta, sodium_n, creatinine_last, bun_last, chloride_last, potassium_last, bicarbonate_last, glucose_last, heart_rate_last, mbp_last, resp_rate_last, spo2_last, temperature_last, urineoutput_total, uo_mlkghr_6hr_last, uo_mlkghr_12hr_last, invasive_vent_pre_t0, noninvasive_vent_pre_t0, hfnc_pre_t0, supplemental_oxygen_pre_t0, tracheostomy_pre_t0, ventilation_no_record_pre_t0, loop_diuretic_pre_t0, vasoactive_pre_t0, rrt_pre_t0, iv_isotonic_crystalloid_pre_t0, dextrose_or_hypotonic_fluid_pre_t0 FROM hypernatremia_feature_base_v01 WHERE is_strict_primary ORDER BY stay_id) TO 'data/strict_primary_model_dataset_v0.1.csv' CSV HEADER
\copy (SELECT subject_id, hadm_id, stay_id, anchor_year_group, temporal_split, outcome_na_ge_151, mannitol_pre_t0, age, gender, sodium_first, sodium_last, sodium_min, sodium_max, sodium_delta, sodium_n, creatinine_last, bun_last, chloride_last, potassium_last, bicarbonate_last, glucose_last, heart_rate_last, mbp_last, resp_rate_last, spo2_last, temperature_last, urineoutput_total, uo_mlkghr_6hr_last, uo_mlkghr_12hr_last, invasive_vent_pre_t0, noninvasive_vent_pre_t0, hfnc_pre_t0, supplemental_oxygen_pre_t0, tracheostomy_pre_t0, ventilation_no_record_pre_t0, loop_diuretic_pre_t0, vasoactive_pre_t0, rrt_pre_t0, iv_isotonic_crystalloid_pre_t0, dextrose_or_hypotonic_fluid_pre_t0 FROM hypernatremia_feature_base_v01 ORDER BY stay_id) TO 'data/loose_sensitivity_model_dataset_v0.1.csv' CSV HEADER
