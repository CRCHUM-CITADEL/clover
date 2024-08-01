import pandas as pd  # 3rd party packages


def convert_precision(
    df_ref: pd.DataFrame, df_to_trans: pd.DataFrame, cont_col: list = None
) -> pd.DataFrame:
    """
    Convert the continuous variables of a dataframe to the same decimal place as its reference variable

    :param df_ref: the reference dataframe
    :param df_to_trans: the dataframe to be transformed
    :param cont_col: the continuous variables (must exist in both dataframes)
    :return: the transformed dataframes
    """
    for col in cont_col:
        precision = (
            df_ref[col]
            .apply(lambda x: len(str(x).split(".")[-1]) if isinstance(x, float) else 0)
            .max()
        )
        df_to_trans[col] = df_to_trans[col].apply(
            lambda x: round(x, precision) if isinstance(x, float) else x
        )

        if df_ref[col].dtype == "int":
            df_to_trans[col] = df_to_trans[col].astype(int)

    return df_to_trans
