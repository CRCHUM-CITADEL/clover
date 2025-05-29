import pandas as pd  # 3rd party packages


def transform_data(
    df_ref: pd.DataFrame, df_to_trans: pd.DataFrame, cont_col: list = None
) -> pd.DataFrame:
    """
    Transform the target dataframe:

    * Set the columns order to be same as the reference dataframe
    * Convert the continuous variables to the same decimal place as its reference variable
    * Convert the type of data of each column according to the type of the reference dataframe

    :param df_ref: the reference dataframe
    :param df_to_trans: the dataframe to be transformed
    :param cont_col: the continuous variables (must exist in both dataframes)
    :return: the transformed dataframes
    """

    re_col_order = df_ref.columns
    ref_dtypes = df_ref.dtypes.to_dict()

    for col in cont_col:
        precision = (
            df_ref[col]
            .apply(lambda x: len(str(x).split(".")[-1]) if isinstance(x, float) else 0)
            .max()
        )
        df_to_trans[col] = df_to_trans[col].apply(
            lambda x: round(x, precision) if isinstance(x, float) else x
        )

    df_to_trans = df_to_trans.astype(ref_dtypes)
    df_to_trans = df_to_trans[re_col_order]

    return df_to_trans
