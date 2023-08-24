# Standard library
from typing import Union
from pathlib import Path
import tempfile
import os
import json

# 3rd party packages
import pandas as pd
import numpy as np

# Local
from .base import Generator
from .external.tab_ddpm.scripts.train import train as tabddpm_train
from .external.tab_ddpm.scripts.sample import sample as tabddpm_sample
from .external.tab_ddpm import lib as tabddpm_lib
import utils.standard as ustandard


class TabDDPMGenerator(Generator):
    """
    Wrapper of the tabular diffusion models TabDDPM https://github.com/yandex-research/tab-ddpm.

    :cvar name: the name of the generator
    :vartype name: str

    :param df: the data to synthesize
    :param metadata: a dictionary containing the list of **continuous** and **categorical** variables
    :param random_state: for reproducibility purposes
    :param generator_filepath: the path of the generator to sample from if it exists
    :param learning_rate: the learning rate for training
    :param batch_size: the batch size for training and sampling
    :param num_timesteps: the diffusion timesteps for the forward diffusion process
    :param num_iter: the training iterations
    :param layers: the width of the MLP layers
    """

    name = "TabDDPM"

    def __init__(
        self,
        df: pd.DataFrame,
        metadata: dict,
        random_state: int = None,
        generator_filepath: Union[Path, str] = None,
        learning_rate: int = 1e-5,
        batch_size: int = 256,
        num_timesteps: int = 100,
        num_iter: int = 1000,
        layers: list[int] = None,
    ):
        super().__init__(df, metadata, random_state, generator_filepath)

        if generator_filepath is None:
            # Load the default configurations
            self._config = tabddpm_lib.load_config(
                Path("./generators/external/tab_ddpm/config.toml")
            )
            if layers is not None:
                self._config["model_params"]["rtdl_params"]["d_layers"] = layers
            self._config["diffusion_params"]["num_timesteps"] = num_timesteps
            self._config["train"]["main"]["steps"] = num_iter
            self._config["train"]["main"]["lr"] = learning_rate
            self._config["train"]["main"]["batch_size"] = batch_size
            self._config["seed"] = random_state
            self._config["train"]["T"]["seed"] = random_state
            self._config["sample"]["seed"] = random_state
        else:  # load the saved configurations
            self._config = tabddpm_lib.load_config(
                Path(generator_filepath).parent / "config.toml"
            )

        # Needed for the wrapping since the data need to be splitted into numerical, categorical and y and saved
        self._tmp_real_data_path = tempfile.TemporaryDirectory()
        self._order_num_cat_y_columns = None

    def __del__(self):
        # Delete the temporary folder containing the preprocessed data
        self._tmp_real_data_path.cleanup()

    def preprocess(self) -> None:
        """
        Prepare the parameters to train the generator.

        :return: *None*
        """

        # Variable to predict
        self._config["model_params"]["is_y_cond"] = (
            self._metadata["variable_to_predict"] in self._metadata["categorical"]
        )
        if self._config["model_params"]["is_y_cond"]:
            self._config["model_params"]["num_classes"] = self._df[
                self._metadata["variable_to_predict"]
            ].nunique()

        # Data split (files need to be temporary saved since TadDDPM ask for a path)
        self._config["real_data_path"] = Path(self._tmp_real_data_path.name)

        #   Remove the variable to predict from list of columns
        continuous_cols = self._metadata["continuous"]
        categorical_cols = self._metadata["categorical"]
        if self._config["model_params"]["is_y_cond"]:
            categorical_cols = [
                col
                for col in categorical_cols
                if col != self._metadata["variable_to_predict"]
            ]
        else:
            continuous_cols = [
                col
                for col in continuous_cols
                if col != self._metadata["variable_to_predict"]
            ]
        self._order_num_cat_y_columns = (
            continuous_cols + categorical_cols + [self._metadata["variable_to_predict"]]
        )

        #   Split X and y and continuous/categorical
        X_cont = self._df[continuous_cols].to_numpy()
        X_cat = self._df[categorical_cols].to_numpy()
        y = self._df[self._metadata["variable_to_predict"]].to_numpy()

        #   Save as numpy files
        np.save(self._config["real_data_path"] / "X_num_train.npy", X_cont)
        np.save(self._config["real_data_path"] / "X_cat_train.npy", X_cat)
        np.save(self._config["real_data_path"] / "y_train.npy", y)

        # Save fake files for val and test splits (not needed here)
        for split in ["val", "test"]:
            df_sample = self._df.sample(frac=1, replace=True)
            np.save(
                self._config["real_data_path"] / f"X_num_{split}.npy",
                df_sample[continuous_cols].to_numpy(),
            )
            np.save(
                self._config["real_data_path"] / f"X_cat_{split}.npy",
                df_sample[categorical_cols].to_numpy(),
            )
            np.save(
                self._config["real_data_path"] / f"y_{split}.npy",
                df_sample[self._metadata["variable_to_predict"]].to_numpy(),
            )

        # Features
        self._config["num_numerical_features"] = len(continuous_cols)

        # info.json
        if self._config["model_params"]["num_classes"] == 0:
            task_type = "regression"
        elif self._config["model_params"]["num_classes"] == 2:
            task_type = "binclass"
        else:
            task_type = "multiclass"
        info = {
            "name": "Dataset",
            "id": "id",
            "task_type": task_type,
            "n_num_features": len(continuous_cols),
            "n_cat_features": len(categorical_cols),
            "test_size": len(X_cont),
            "train_size": len(X_cont),
            "val_size": len(X_cont),
        }
        with open(self._config["real_data_path"] / "info.json", "w") as f:
            json.dump(info, f)

    def fit(self, save_path: Union[Path, str]) -> None:
        """
        Construct the sequential trees.

        :param save_path: the path to save the generator
        :return: *None*
        """

        self._config["parent_dir"] = Path(save_path)

        with ustandard.HiddenPrints():
            tabddpm_train(
                **self._config["train"]["main"],
                **self._config["diffusion_params"],
                parent_dir=self._config["parent_dir"],
                real_data_path=self._config["real_data_path"],
                model_type=self._config["model_type"],
                model_params=self._config["model_params"],
                T_dict=self._config["train"]["T"],
                num_numerical_features=self._config["num_numerical_features"],
                device=self._config["device"],
                seed=self._config["seed"],
                change_val=False,
            )

        ustandard.save_pickle(
            obj=self._gen,
            folderpath=save_path,
            filename=TabDDPMGenerator.name,
            date=True,
        )

    def display(self) -> None:
        """
        Print the parameters of TabDDPM.

        :return: *None*
        """
        print("Generator: TabDDPM")
        print("Parameters: ", self._config)

    def sample(self, save_path: Union[Path, str], num_samples: int = 1) -> pd.DataFrame:
        """
        Generate samples using the sequential trees trained on the real data.

        :param save_path: the path to save the generated samples
        :param num_samples: the number of samples to generate
        :return: the generated samples
        """

        with ustandard.HiddenPrints():
            tabddpm_sample(
                num_samples=num_samples,
                batch_size=self._config["train"]["main"]["batch_size"],
                disbalance=self._config["sample"].get("disbalance", None),
                **self._config["diffusion_params"],
                parent_dir=self._config["parent_dir"],
                real_data_path=self._config["real_data_path"],
                model_path=os.path.join(self._config["parent_dir"], "model.pt"),
                model_type=self._config["model_type"],
                model_params=self._config["model_params"],
                T_dict=self._config["train"]["T"],
                num_numerical_features=self._config["num_numerical_features"],
                device=self._config["device"],
                seed=self._config["sample"]["seed"],
                change_val=False,
            )

        # Load the generated samples
        X_num_generated = np.load(
            self._config["parent_dir"] / "X_num_train.npy", allow_pickle=True
        )
        X_cat_generated = np.load(
            self._config["parent_dir"] / "X_cat_train.npy", allow_pickle=True
        )
        y_generated = np.load(
            self._config["parent_dir"] / "y_train.npy", allow_pickle=True
        )

        # Rebuild the dataframe
        samples = np.concatenate(
            (X_num_generated, X_cat_generated, y_generated[:, None]), axis=1
        )
        samples = pd.DataFrame(samples, columns=self._order_num_cat_y_columns)
        samples = samples[self._df.columns]  # same initial columns order
        samples = samples.astype(self._df.dtypes.to_dict())

        samples.to_csv(
            Path(save_path)
            / f"{ustandard.get_date()}_{TabDDPMGenerator.name}_{num_samples}samples.csv",
            index=False,
        )

        # Delete the generated split samples
        os.remove(self._config["parent_dir"] / "X_num_train.npy")
        os.remove(self._config["parent_dir"] / "X_cat_train.npy")
        os.remove(self._config["parent_dir"] / "y_train.npy")

        return samples
