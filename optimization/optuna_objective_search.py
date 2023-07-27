# Standard library
import tempfile

from ray.air import session
import pandas as pd

from generators.smote import SmoteGenerator
from generators.tvae_generator import TVAEGenerator
from optimization.objective_function import distinguishability_hinge_loss
from utils import standard as ustandard

generators_mapping = {"SMOTE": SmoteGenerator, "TVAE": TVAEGenerator}

objective_function_mapping = {
    "distinguishability_hinge_loss": distinguishability_hinge_loss
}


def distinguishability_objective_function(config, data):
    """

    :param config:
    :param data:
    :return:
    """

    df = pd.read_csv(data["df_path"])
    metadata = ustandard.load_pickle(data["metadata_path"])

    gen = generators_mapping[data["generator"]](df=df, metadata=metadata, **config)
    gen.preprocess()

    with tempfile.TemporaryDirectory() as temp_dir:  # no need to keep the generated samples
        gen.fit(save_path=temp_dir)
        df_synth = gen.sample(save_path=temp_dir, num_samples=len(df))

    cost = objective_function_mapping[data["objective_function"]](
        df=df,
        df_to_compare=df_synth,
        metadata=metadata,
        minimize=True,
        use_gpu=data["use_gpu"],
    )

    # session.report(metrics={"score": cost})
    return cost
