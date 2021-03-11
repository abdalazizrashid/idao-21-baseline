import configparser
import gc
import logging
import pathlib as path
import sys
from collections import defaultdict
from itertools import chain
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikitplot as skplt
import torch
from more_itertools import bucket

from idao.data_module import IDAODataModule
from idao.model import SimpleConv
from idao.utils import delong_roc_variance



dict_pred = defaultdict(list)

def make_csv(mode, dataloader, checkpoint_path, cfg):
    torch.multiprocessing.set_sharing_strategy("file_system")
    logging.info("Loading checkpoint")
    model = SimpleConv.load_from_checkpoint(checkpoint_path, mode=mode)
    model = model.cpu().eval()

    if mode == "classification":
        logging.info("Classification model loaded")
    else:
        logging.info("Regression model loaded")

    for _, (img, name) in enumerate(iter(dataloader)):
        if mode == "classification":
            dict_pred["id"].append(name[0].split('.')[0])
            output = (1 if torch.round(model(img)["class"].detach()[0][0]) == 0 else 0)
            dict_pred["classification_predictions"].append(output)

        else:
            output = model(img)["energy"].detach()
            dict_pred["regression_predictions"].append(output[0][0].item())


def main(cfg):
    PATH = path.Path(cfg["DATA"]["DatasetPath"])

    dataset_dm = IDAODataModule(
        data_dir=PATH, batch_size=64, cfg=cfg
    )

    dataset_dm.prepare_data()
    dataset_dm.setup()
    dl = dataset_dm.test_dataloader()

    for mode in ["regression", "classification"]:
        if mode == "classification":
            model_path = cfg["REPORT"]["ClassificationCheckpoint"]
        else:
            model_path = cfg["REPORT"]["RegressionCheckpoint"]

        make_csv(mode, dl, model_path, cfg=cfg)

    data_frame = pd.DataFrame(dict_pred, columns=["id", "classification_predictions", "regression_predictions"])
    data_frame.to_csv('submission.csv', index=False, header=True)


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("./config.ini")
    main(cfg=config)
