import configparser
import pathlib as path

import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from idao.data_module import IDAODataModule
from idao.model import SimpleConv

seed_everything(666)


def trainer(mode: ["classification", "regression"], cfg):
    # init model
    model = SimpleConv(mode=mode)
    if mode == "classification":
        epochs = cfg["TRAINING"]["ClassificationEpochs"]
    else:
        epochs = cfg["TRAINING"]["RegressionEpochs"]
    # Initialize a trainer
    trainer = pl.Trainer(
        gpus=int(cfg["TRAINING"]["NumGPUs"]),
        max_epochs=int(epochs),
        progress_bar_refresh_rate=20,
        weights_save_path=path.Path(cfg["TRAINING"]["ModelParamsSavePath"]).joinpath(
            mode
        ),
        default_root_dir=path.Path(cfg["TRAINING"]["ModelParamsSavePath"]),
    )

    # Train the model ⚡
    trainer.fit(model, dataset_dm)


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("./config.ini")

    PATH = path.Path(config["DATA"]["DatasetPath"])

    dataset_dm = IDAODataModule(
        data_dir=PATH, batch_size=int(config["TRAINING"]["BatchSize"]), cfg=config
    )
    dataset_dm.prepare_data()
    dataset_dm.setup()

    for mode in ["classification", "regression"]:
        trainer(mode, cfg=config)
