from __future__ import print_function

import warnings

warnings.filterwarnings("ignore")

import argparse
from collections import OrderedDict

import mp_utils as mp
import numpy as np
import pandas as pd
import psycopg2
import sqlalchemy


def main(
    port,
    host,
    sqluser,
    sqlpass,
    sslcert,
    sslkey,
    dbname,
    schema_name,
    study_name,
    output_path,
) -> None:
    query_schema = "SET search_path to public," + schema_name + ";"

    # Connect to local postgres version of mimic
    url = (
        "postgresql+psycopg2://"
        + sqluser
        + ":"
        + sqlpass
        + "@"
        + host
        + ":"
        + port
        + "/"
        + dbname
    )
    engine = sqlalchemy.create_engine(
        url, connect_args={"sslcert": sslcert, "sslkey": sslkey}
    )
    con = psycopg2.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=sqluser,
        sslkey=sslkey,
        sslcert=sslcert,
    )

    # exclusion criteria:
    #   - less than 15 years old
    #   - stayed in the ICU less than 4 hours
    #   - never have any chartevents data (i.e. likely administrative error)
    #   - organ donor accounts (administrative "readmissions" for patients who died in hospital)
    query = (
        query_schema
        + """
    select 
        *
    from dm_cohort
    """
    )
    co = pd.read_sql_query(query, con)  # 61532 unique icu stay

    # convert the inclusion flags to boolean
    for c in co.columns:
        if c[0:10] == "inclusion_":
            co[c] = co[c].astype(bool)

    # extract static vars into a separate dataframe
    df_static = pd.read_sql_query(
        query_schema + "select * from mp_static_data", con
    )  # 52050 unique icu stay

    vars_static = [
        "is_male",
        "emergency_admission",
        "age",
        # services
        "service_any_noncard_surg",
        "service_any_card_surg",
        "service_cmed",
        "service_traum",
        "service_nmed",
        # ethnicities
        "race_black",
        "race_hispanic",
        "race_asian",
        "race_other",
        # phatness
        "height",
        "weight",
        "bmi",
    ]

    # get ~5 million rows containing data from errbody
    # this takes a little bit of time to load into memory (~2 minutes)

    # %%time results
    # CPU times: user 42.8 s, sys: 1min 3s, total: 1min 46s
    # Wall time: 2min 7s

    df = pd.read_sql_query(
        query_schema + "select * from mp_data", con
    )  # heartrate etc. taken at different hour
    df.drop("subject_id", axis=1, inplace=True)
    df.drop("hadm_id", axis=1, inplace=True)
    df.sort_values(["icustay_id", "hr"], axis=0, ascending=True, inplace=True)

    # get death information
    df_death = pd.read_sql_query(
        query_schema
        + """
    select 
    co.subject_id, co.hadm_id, co.icustay_id
    , ceil(extract(epoch from (co.outtime - co.intime))/60.0/60.0) as dischtime_hours
    , ceil(extract(epoch from (adm.deathtime - co.intime))/60.0/60.0) as deathtime_hours
    , case when adm.deathtime is null then 0 else 1 end as death
    from dm_cohort co
    inner join admissions adm
    on co.hadm_id = adm.hadm_id
    where co.excluded = 0
    """,
        con,
    )

    # get censoring information
    df_censor = pd.read_sql_query(
        query_schema
        + """
    select co.icustay_id, min(cs.charttime) as censortime
    , ceil(extract(epoch from min(cs.charttime-co.intime) )/60.0/60.0) as censortime_hours
    from dm_cohort co 
    inner join mp_code_status cs
    on co.icustay_id = cs.icustay_id
    where cmo+dnr+dni+dncpr+cmo_notes>0
    and co.excluded = 0
    group by co.icustay_id
    """,
        con,
    )

    # print out the exclusions *SEQUENTIALLY* - i.e. if already excluded, don't re-print
    print("")
    # print("====================={}==========".format("=" * len(study_name)))
    # print("Cohort - initial size: {} ICU stays".format(co.shape[0]))

    # Base exclusion criteria: exclude < 15 yrs old, organ donner etc.
    idxRem = np.zeros(co.shape[0], dtype=bool)
    for c in co.columns:
        if c[0 : len("exclusion_")] == "exclusion_":
            # N_REM = np.sum((co[c].values == 1))
            # print(
            #     "  {:5g} ({:2.2f}%) - {}".format(N_REM, N_REM * 100.0 / co.shape[0], c)
            # )
            idxRem[co[c].values == 1] = True

    # summarize all exclusions
    # N_REM = np.sum(idxRem)
    # print(
    #     "  {:5g} ({:2.2f}%) - {}".format(
    #         N_REM, N_REM * 100.0 / co.shape[0], "all exclusions"
    #     )
    # )
    # print("")
    # print(
    #     "Final cohort size: {} ICU stays ({:2.2f}%).".format(
    #         co.shape[0] - np.sum(idxRem), (1 - np.mean(idxRem)) * 100.0
    #     )
    # )
    co = co.loc[~idxRem, :]

    # Exclusion criteria
    """
    Each study has its own exclusion criteria (sometimes studies have multiple experiments). 
    We define a dictionary of all exclusions with the dictionary key as the study name. 
    Some studies have multiple experiments, so we append *a*, *b*, or *c*.
    
    The dictionary stores a length 2 list. 
    The first element defines the window for data extraction: 
    it contains a dictionary of the windows and the corresponding window sizes. 
    The second element is the exclusion criteria. Both are functions which use `co` or `df` as their input.
    """

    # first we can define the different windows: there aren't that many!
    df_tmp = co.copy().set_index("icustay_id")

    # admission+12 hours; window end at 12 hours after ICU admission
    time_12hr = df_tmp.copy()
    time_12hr["windowtime"] = 12
    time_12hr = time_12hr["windowtime"].to_dict()

    # admission+24 hours
    time_24hr = df_tmp.copy()
    time_24hr["windowtime"] = 24
    time_24hr = time_24hr["windowtime"].to_dict()

    # admission+48 hours
    time_48hr = df_tmp.copy()
    time_48hr["windowtime"] = 48
    time_48hr = time_48hr["windowtime"].to_dict()

    # admission+72 hours
    time_72hr = df_tmp.copy()
    time_72hr["windowtime"] = 72
    time_72hr = time_72hr["windowtime"].to_dict()

    # admission+96 hours
    time_96hr = df_tmp.copy()
    time_96hr["windowtime"] = 96
    time_96hr = time_96hr["windowtime"].to_dict()

    # entire stay
    time_all = df_tmp.copy()
    time_all = time_all["dischtime_hours"].apply(np.ceil).astype(int).to_dict()

    # 12 hours before the patient died/discharged
    time_predeath = df_tmp.copy()
    time_predeath["windowtime"] = time_predeath["dischtime_hours"]
    idx = time_predeath["deathtime_hours"] < time_predeath["dischtime_hours"]
    time_predeath.loc[idx, "windowtime"] = time_predeath.loc[idx, "deathtime_hours"]
    # move from discharge/death time to 12 hours beforehand
    time_predeath["windowtime"] = time_predeath["windowtime"] - 12
    time_predeath = time_predeath["windowtime"].apply(np.ceil).astype(int).to_dict()

    # example params used to extract patient data
    # element 1: dictionary specifying end time of window for each patient
    # element 2: size of window
    # element 3: extra hours added to make it easier to get data on labs (and allows us to get labs pre-ICU)
    # e.g. [time_24hr, 8, 24] is
    #   (1) window ends at admission+24hr
    #   (2) window is 8 hours long
    #   (3) lab window is 8+24=32 hours long

    # inclusion_ cols have been converted to bool type
    def inclFcn(x, inclusions):
        return x.loc[x[inclusions].all(axis=1), "icustay_id"]

    # this one is used more than once, so we define it here
    hugExclFcnMIMIC3 = lambda x: x.loc[
        x["inclusion_over_18"]
        & x["inclusion_hug2009_obs"]
        & x["inclusion_hug2009_not_nsicu_csicu"]
        & x["inclusion_first_admission"]
        & x["inclusion_full_code"]
        & x["inclusion_not_brain_death"]
        & x["inclusion_not_crf"],
        "icustay_id",
    ].values
    hugExclFcn = lambda x: np.intersect1d(
        hugExclFcnMIMIC3(x), x.loc[x["inclusion_only_mimicii"], "icustay_id"].values
    )

    # physionet2012 subset - not exact but close
    def physChallExclFcn(x):
        out = x.loc[
            x["inclusion_only_mimicii"]
            & x["inclusion_over_18"]
            & x["inclusion_stay_ge_48hr"]
            & x["inclusion_has_saps"],
            "icustay_id",
        ].values
        out = np.sort(out)
        out = out[0:4000]
        return out

    # caballero2015 is a random subsample - then limits to 18yrs, resulting in 11648
    def caballeroExclFcn(x):
        out = x.loc[
            x["inclusion_only_mimicii"] & x["inclusion_over_18"], "icustay_id"
        ].values
        out = np.sort(out)
        out = out[0:11648]
        return out

    np.random.seed(546345)
    W_extra = 24

    exclusions = OrderedDict(
        [
            [
                "caballero2015dynamically_a",
                [[time_24hr, 24, W_extra], caballeroExclFcn, "hospital_expire_flag"],
            ],
            [
                "caballero2015dynamically_b",
                [[time_48hr, 48, W_extra], caballeroExclFcn, "hospital_expire_flag"],
            ],
            [
                "caballero2015dynamically_c",
                [[time_72hr, 72, W_extra], caballeroExclFcn, "hospital_expire_flag"],
            ],
            [
                "calvert2016computational",
                [
                    [time_predeath, 5, W_extra],
                    lambda x: x.loc[
                        x["inclusion_over_18"]
                        & x["inclusion_only_micu"]
                        & x["inclusion_calvert2016_obs"]
                        & x["inclusion_stay_ge_17hr"]
                        & x["inclusion_stay_le_500hr"]
                        & x["inclusion_non_alc_icd9"],
                        "icustay_id",
                    ].values,
                    "hospital_expire_flag",
                ],
            ],
            [
                "calvert2016using",
                [
                    [time_predeath, 5, W_extra],
                    lambda x: x.loc[
                        x["inclusion_over_18"]
                        & x["inclusion_only_micu"]
                        & x["inclusion_calvert2016_obs"]
                        & x["inclusion_stay_ge_17hr"]
                        & x["inclusion_stay_le_500hr"],
                        "icustay_id",
                    ].values,
                    "hospital_expire_flag",
                ],
            ],
            [
                "celi2012database_a",
                [
                    [time_72hr, 72, W_extra],
                    lambda x: x.loc[
                        x["inclusion_only_mimicii"]
                        & x["inclusion_over_18"]
                        & x["inclusion_aki_icd9"],
                        "icustay_id",
                    ].values,
                    "hospital_expire_flag",
                ],
            ],
            [
                "celi2012database_b",
                [
                    [time_24hr, 24, W_extra],
                    lambda x: x.loc[
                        x["inclusion_only_mimicii"]
                        & x["inclusion_over_18"]
                        & x["inclusion_sah_icd9"],
                        "icustay_id",
                    ].values,
                    "hospital_expire_flag",
                ],
            ],
            [
                "che2016recurrent_a",
                [
                    [time_48hr, 48, W_extra],
                    lambda x: x.loc[x["inclusion_over_18"], "icustay_id"].values,
                    "death_48hr_post_icu_admit",
                ],
            ],
            [
                "che2016recurrent_b",
                [[time_48hr, 48, W_extra], physChallExclFcn, "hospital_expire_flag"],
            ],
            [
                "ding2016mortality",
                [[time_48hr, 48, W_extra], physChallExclFcn, "hospital_expire_flag"],
            ],
            [
                "ghassemi2014unfolding_a",
                [
                    [time_24hr, 24, W_extra],
                    lambda x: x.loc[
                        x["inclusion_only_mimicii"]
                        & x["inclusion_over_18"]
                        & x["inclusion_ge_100_non_stop_words"]
                        & x["inclusion_stay_ge_24hr"],
                        "icustay_id",
                    ].values,
                    "hospital_expire_flag",
                ],
            ],
            [
                "ghassemi2014unfolding_b",
                [
                    [time_12hr, 12, W_extra],
                    lambda x: x.loc[
                        x["inclusion_only_mimicii"]
                        & x["inclusion_over_18"]
                        & x["inclusion_ge_100_non_stop_words"]
                        & x["inclusion_stay_ge_12hr"],
                        "icustay_id",
                    ].values,
                    "hospital_expire_flag",
                ],
            ],
            [
                "ghassemi2014unfolding_c",
                [
                    [time_12hr, 12, W_extra],
                    lambda x: x.loc[
                        x["inclusion_only_mimicii"]
                        & x["inclusion_over_18"]
                        & x["inclusion_ge_100_non_stop_words"]
                        & x["inclusion_stay_ge_12hr"],
                        "icustay_id",
                    ].values,
                    "death_30dy_post_hos_disch",
                ],
            ],
            [
                "ghassemi2014unfolding_d",
                [
                    [time_12hr, 12, W_extra],
                    lambda x: x.loc[
                        x["inclusion_only_mimicii"]
                        & x["inclusion_over_18"]
                        & x["inclusion_ge_100_non_stop_words"]
                        & x["inclusion_stay_ge_12hr"],
                        "icustay_id",
                    ].values,
                    "death_1yr_post_hos_disch",
                ],
            ],
            [
                "ghassemi2015multivariate_a",
                [
                    [time_24hr, 24, W_extra],
                    lambda x: x.loc[
                        x["inclusion_only_mimicii"]
                        & x["inclusion_over_18"]
                        & x["inclusion_ge_100_non_stop_words"]
                        & x["inclusion_gt_6_notes"]
                        & x["inclusion_stay_ge_24hr"]
                        & x["inclusion_has_saps"],
                        "icustay_id",
                    ].values,
                    "hospital_expire_flag",
                ],
            ],
            [
                "ghassemi2015multivariate_b",
                [
                    [time_24hr, 24, W_extra],
                    lambda x: x.loc[
                        x["inclusion_only_mimicii"]
                        & x["inclusion_over_18"]
                        & x["inclusion_ge_100_non_stop_words"]
                        & x["inclusion_gt_6_notes"]
                        & x["inclusion_stay_ge_24hr"]
                        & x["inclusion_has_saps"],
                        "icustay_id",
                    ].values,
                    "death_1yr_post_hos_disch",
                ],
            ],
            [
                "grnarova2016neural_a",
                [
                    [time_all, 24, W_extra],
                    lambda x: x.loc[
                        x["inclusion_over_18"] & x["inclusion_multiple_hadm"],
                        "icustay_id",
                    ].values,
                    "hospital_expire_flag",
                ],
            ],
            [
                "grnarova2016neural_b",
                [
                    [time_all, 24, W_extra],
                    lambda x: x.loc[
                        x["inclusion_over_18"] & x["inclusion_multiple_hadm"],
                        "icustay_id",
                    ].values,
                    "death_30dy_post_hos_disch",
                ],
            ],
            [
                "grnarova2016neural_c",
                [
                    [time_all, 24, W_extra],
                    lambda x: x.loc[
                        x["inclusion_over_18"] & x["inclusion_multiple_hadm"],
                        "icustay_id",
                    ].values,
                    "death_1yr_post_hos_disch",
                ],
            ],
            [
                "harutyunyan2017multitask",
                [
                    [time_48hr, 48, W_extra],
                    lambda x: x.loc[
                        x["inclusion_over_18"] & x["inclusion_multiple_icustay"],
                        "icustay_id",
                    ].values,
                    "hospital_expire_flag",
                ],
            ],  ## This one contains the most data
            [
                "hoogendoorn2016prediction",
                [
                    [time_24hr, 24, W_extra],
                    lambda x: x.loc[
                        x["inclusion_only_mimicii"]
                        & x["inclusion_over_18"]
                        & x["inclusion_hug2009_obs"]
                        & x["inclusion_stay_ge_24hr"],
                        "icustay_id",
                    ].values,
                    "hospital_expire_flag",
                ],
            ],
            [
                "hug2009icu",
                [[time_24hr, 24, W_extra], hugExclFcn, "death_30dy_post_icu_disch"],
            ],
            [
                "johnson2012patient",
                [[time_48hr, 48, W_extra], physChallExclFcn, "hospital_expire_flag"],
            ],
            [
                "johnson2014data",
                [[time_48hr, 48, W_extra], physChallExclFcn, "hospital_expire_flag"],
            ],
            [
                "joshi2012prognostic",
                [[time_24hr, 24, W_extra], hugExclFcn, "hospital_expire_flag"],
            ],
            [
                "joshi2016identifiable",
                [
                    [time_48hr, 48, W_extra],
                    lambda x: x.loc[
                        x["inclusion_over_18"] & x["inclusion_stay_ge_48hr"],
                        "icustay_id",
                    ].values,
                    "hospital_expire_flag",
                ],
            ],
            [
                "lee2015customization_a",
                [
                    [time_24hr, 24, W_extra],
                    lambda x: x.loc[
                        x["inclusion_only_mimicii"]
                        & x["inclusion_over_18"]
                        & x["inclusion_lee2015_service"]
                        & x["inclusion_has_saps"]
                        & x["inclusion_stay_ge_24hr"],
                        "icustay_id",
                    ].values,
                    "hospital_expire_flag",
                ],
            ],
            [
                "lee2015customization_b",
                [
                    [time_24hr, 24, W_extra],
                    lambda x: x.loc[
                        x["inclusion_only_mimicii"]
                        & x["inclusion_over_18"]
                        & x["inclusion_lee2015_service"]
                        & x["inclusion_has_saps"]
                        & x["inclusion_stay_ge_24hr"],
                        "icustay_id",
                    ].values,
                    "death_30dy_post_hos_disch",
                ],
            ],
            [
                "lee2015customization_c",
                [
                    [time_24hr, 24, W_extra],
                    lambda x: x.loc[
                        x["inclusion_only_mimicii"]
                        & x["inclusion_over_18"]
                        & x["inclusion_lee2015_service"]
                        & x["inclusion_has_saps"]
                        & x["inclusion_stay_ge_24hr"],
                        "icustay_id",
                    ].values,
                    "death_2yr_post_hos_disch",
                ],
            ],
            [
                "lee2015personalized",
                [
                    [time_24hr, 24, W_extra],
                    lambda x: x.loc[
                        x["inclusion_only_mimicii"]
                        & x["inclusion_over_18"]
                        & x["inclusion_has_saps"]
                        & x["inclusion_stay_ge_24hr"],
                        "icustay_id",
                    ].values,
                    "death_30dy_post_hos_disch",
                ],
            ],
            [
                "lee2017patient",
                [
                    [time_24hr, 24, W_extra],
                    lambda x: x.loc[
                        x["inclusion_only_mimicii"]
                        & x["inclusion_over_18"]
                        & x["inclusion_has_saps"]
                        & x["inclusion_stay_ge_24hr"],
                        "icustay_id",
                    ].values,
                    "death_30dy_post_hos_disch",
                ],
            ],
            [
                "lehman2012risk",
                [
                    [time_24hr, 24, W_extra],
                    lambda x: x.loc[
                        x["inclusion_only_mimicii"]
                        & x["inclusion_over_18"]
                        & x["inclusion_has_saps"]
                        & x["inclusion_stay_ge_24hr"]
                        & x["inclusion_first_admission"],
                        "icustay_id",
                    ].values,
                    "hospital_expire_flag",
                ],
            ],
            [
                "luo2016interpretable_a",
                [
                    [time_all, 24, W_extra],
                    lambda x: x.loc[
                        x["inclusion_only_mimicii"]
                        & x["inclusion_over_18"]
                        & x["inclusion_has_sapsii"]
                        & x["inclusion_no_disch_summary"],
                        "icustay_id",
                    ].values,
                    "death_30dy_post_hos_disch",
                ],
            ],
            [
                "luo2016interpretable_b",
                [
                    [time_all, 24, W_extra],
                    lambda x: x.loc[
                        x["inclusion_only_mimicii"]
                        & x["inclusion_over_18"]
                        & x["inclusion_has_sapsii"]
                        & x["inclusion_no_disch_summary"],
                        "icustay_id",
                    ].values,
                    "death_6mo_post_hos_disch",
                ],
            ],
            [
                "luo2016predicting",
                [
                    [time_24hr, 12, W_extra],
                    lambda x: np.intersect1d(
                        hugExclFcn(x),
                        x.loc[x["inclusion_stay_ge_24hr"], "icustay_id"].values,
                    ),
                    "death_30dy_post_icu_disch",
                ],
            ],
            [
                "pirracchio2015mortality",
                [
                    [time_24hr, 24, W_extra],
                    lambda x: x.loc[x["inclusion_only_mimicii"], "icustay_id"].values,
                    "hospital_expire_flag",
                ],
            ],
            [
                "ripoll2014sepsis",
                [
                    [time_24hr, 24, W_extra],
                    lambda x: x.loc[
                        x["inclusion_only_mimicii"]
                        & x["inclusion_over_18"]
                        & x["inclusion_has_saps"]
                        & x["inclusion_not_explicit_sepsis"],
                        "icustay_id",
                    ].values,
                    "hospital_expire_flag",
                ],
            ],
            [
                "wojtusiak2017c",
                [
                    [time_all, 24, W_extra],
                    lambda x: x.loc[
                        x["inclusion_over_65"] & x["inclusion_alive_hos_disch"],
                        "icustay_id",
                    ].values,
                    "death_30dy_post_hos_disch",
                ],
            ],
        ]
    )

    # define var_static which is used later
    (
        var_min,
        var_max,
        var_first,
        var_last,
        var_sum,
        var_first_early,
        var_last_early,
        var_static,
    ) = mp.vars_of_interest()

    print("")
    print("====================={}==========".format("=" * len(study_name)))
    print("=========== STUDY: {}============".format(study_name))
    print("====================={}==========".format("=" * len(study_name)))

    params = exclusions[study_name][0]
    df_data = mp.get_design_matrix(df, params[0], W=params[1], W_extra=params[2])

    # get a list of icustay_id
    iid_keep = exclusions[study_name][1](co)
    print(
        "Reducing sample size from {} to {} ({:2.2f}%)".format(
            df_data.shape[0],
            iid_keep.shape[0],
            iid_keep.shape[0] * 100.0 / df_data.shape[0],
        )
    )
    df_data = df_data.reindex(
        iid_keep
    )  # reproduce pandas < 1 behavior (NAN when index does not exist): df_data = df_data.loc[iid_keep,:]

    y_outcome_label = exclusions[study_name][2]

    # load the data into a numpy array

    # first, the data from static vars from df_static
    X = df_data.merge(
        df_static.set_index("icustay_id")[var_static],
        how="left",
        left_index=True,
        right_index=True,
    )

    # next, add in the outcome: death in hospital
    X = X.merge(
        co.set_index("icustay_id")[[y_outcome_label]], left_index=True, right_index=True
    )

    X.reset_index(inplace=True)
    print(f"Shape: {X.shape}")

    df_count = X.describe().loc["count", :]
    col_to_remove = df_count[df_count <= X.shape[0] * 0.7].index.to_list()

    # remove the entire column if 30% of the data is missing
    df_new = X.drop(col_to_remove, axis=1)

    # remove rows that have missing value
    df_new = df_new.dropna()

    # remove icustay_id
    df_new = df_new.drop("icustay_id", axis=1)

    df_new.to_csv(output_path, index=False)

    print("")
    print("After removing NAs")
    print(f"Shape: {df_new.shape}")
    print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MIMIC dataset")
    parser.add_argument(
        "--port",
        default=None,
        type=str,
        help="Database port",
    )
    parser.add_argument(
        "--host",
        default=None,
        type=str,
        help="Database host",
    )
    parser.add_argument(
        "--sqluser",
        default=None,
        type=str,
        help="User name for database",
    )
    parser.add_argument(
        "--sqlpass",
        default=None,
        type=str,
        help="Password of user",
    )
    parser.add_argument(
        "--sslcert",
        default=None,
        type=str,
        help="Path that points to the SSL certificate",
    )
    parser.add_argument(
        "--sslkey",
        default=None,
        type=str,
        help="Path that points to the SSL key file",
    )
    parser.add_argument(
        "--dbname",
        default=None,
        type=str,
        help="Name of the database",
    )
    parser.add_argument(
        "--schema_name",
        default=None,
        type=str,
        help="Name of the schema",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        type=str,
        help="Path of the saved dataset",
    )

    parser.add_argument(
        "--study_name",
        default=None,
        type=str,
        help="Name of the study",
    )

    args = parser.parse_args()
    main(
        port=args.port,
        host=args.host,
        sqluser=args.sqluser,
        sqlpass=args.sqlpass,
        sslcert=args.sslcert,
        sslkey=args.sslkey,
        dbname=args.dbname,
        schema_name=args.schema_name,
        study_name=args.study_name,
        output_path=args.output_path,
    )
