import configparser
import gc
import logging
import pathlib as path
import sys
from collections import defaultdict
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
import scikitplot as skplt
import torch
from more_itertools import bucket

from idao.data_module import IDAODataModule
from idao.model import SimpleConv
from idao.utils import delong_roc_variance


def test_variance(target, predictions):
    return torch.std(torch.abs(predictions - target) / target) ** 2


def run_test(mode, dataloader, checkpoint_path, cfg):
    torch.multiprocessing.set_sharing_strategy("file_system")
    logging.info("Loading checkpoint")
    model = SimpleConv.load_from_checkpoint(checkpoint_path, mode=mode)
    model = model.cpu().eval()
    regression_predictions = []
    classification_predictions = []
    classification_target = []
    regression_target = []

    if mode == "classification":
        logging.info("Classification model loaded")
    else:
        logging.info("Regression model loaded")

    for i, (img, class_label, regression_label, _) in enumerate(iter(dataloader)):
        if mode == "classification":
            output = model(img)["class"].detach()
            classification_predictions.append(output)
            classification_target.append(class_label)
        else:
            output = model(img)["energy"].detach()
            regression_predictions.append(output)
            regression_target.append(regression_label)

    #        del output

    if mode == "classification":
        logging.info("Starting classification task")
        classification_predictions = torch.cat(classification_predictions, dim=0)
        classification_target = torch.cat(classification_target, dim=0)
        new_target = np.argmax(classification_target.detach().cpu().numpy(), axis=1)
        auc, variance = delong_roc_variance(
            new_target, classification_predictions.detach().cpu().numpy()[:, 1]
        )
        skplt.metrics.plot_roc(
            classification_target.max(1).indices,
            classification_predictions.detach().cpu().numpy(),
            plot_macro=False,
            plot_micro=False,
            classes_to_plot=[0]
        )
        plt.savefig(f'{cfg["REPORT"]["SaveDir"]}/roc_auc.png', dpi=196)
        plt.show()
        logging.info(f'ROC plot saved at: {cfg["REPORT"]["SaveDir"]}/roc_auc.png')
        logging.info(f"Delong => ROC-AUC: {auc} variance: {variance}")

        del classification_predictions
        del classification_target
        # gc.collect() # Invoke the garbage collector
        return (None, auc)

    else:
        logging.info("Starting regression task")
        regression_predictions = torch.tensor(
            list(chain(*regression_predictions))
        ).view(-1)
        regression_target = torch.tensor(list(chain(*regression_target))).view(-1)
        variance = test_variance(regression_target, regression_predictions)
        logging.info(f"Test energy variance: {variance}")

        # MAE
        mae = torch.nn.functional.l1_loss(regression_predictions, regression_target)

        # plot correlation
        fig, ax = plt.subplots()
        ax.plot(
            regression_target, regression_predictions, "ro", label="Energy Prediction"
        )
        ax.set_xlabel("True Energy")
        ax.set_ylabel("Predicted Energy")
        ax.legend()
        ax.grid()
        fig.savefig(f'{cfg["REPORT"]["SaveDir"]}/energy_correlation.png', dpi=196)
        logging.info(f'Energy correlation plot saved at: {cfg["REPORT"]["SaveDir"]}/energy_correlation.png')
        plt.close(fig)

        logging.info(f'===> Length: {len(regression_predictions)} {len(regression_target)}')
        # plot comparison
        fig1, ax1 = plt.subplots()
        ax1.plot(regression_predictions, "bo", alpha=0.6, label="Predicted Energy")
        ax1.plot(regression_target, "ro", label="True Energy")
        ax1.legend()
        ax1.grid()
        fig1.savefig(f'{cfg["REPORT"]["SaveDir"]}/energy_comparison.png', dpi=196)
        logging.info(f'Energy comparison plot saved at: {cfg["REPORT"]["SaveDir"]}/energy_comparison.png')
        plt.close(fig1)

        # plot histograms
        group = zip(regression_target, regression_predictions)
        group = sorted(group, key=lambda item: item[0])
        logging.info(regression_target)

        data_dict = defaultdict(list)

        for t, p in group:
            data_dict[t.item()].append(p.item())

        for i, (k, v) in enumerate(data_dict.items()):
            fig3, ax3 = plt.subplots()
            ax3.hist(
                v,
                bins=100,
                histtype="step",
                label=f"Energy: {k} keV \n RMS: {float(torch.sqrt(torch.mean(torch.tensor(v)**2))):.03f} \n Mean: {float(torch.mean(torch.tensor(v))):.03f}",
            )
            ax3.legend()
            fig3.savefig(f'{cfg["REPORT"]["SaveDir"]}/energy_hist{k}_{i}.png', dpi=196)
            plt.close(fig3)

            logging.info(
                f'Histogram {k} keV saved at: {cfg["REPORT"]["SaveDir"]}/energy_hist{k}_{i}.png'
            )

        return (mae, None)


def main(cfg):
    PATH = path.Path(cfg["DATA"]["DatasetPath"])

    dataset_dm = IDAODataModule(
        data_dir=PATH, batch_size=64, cfg=cfg
    )

    dataset_dm.prepare_data()
    dataset_dm.setup()
    #dl = dataset_dm.test_dataloader()
    dl = dataset_dm.train_dataloader()
    mae = 0
    variance = 0

    for mode in ["regression", "classification"]:
        if mode == "classification":
            model_path = cfg["REPORT"]["ClassificationCheckpoint"]
        else:
            model_path = cfg["REPORT"]["RegressionCheckpoint"]

        _mae, _auc = run_test(mode, dl, model_path, cfg=cfg)
        if _mae is not None:
            mae = _mae
        if _auc is not None:
            auc = _auc

        gc.collect()
    logging.info(f'MAE = {mae}')
    logging.info(f'AUC = {auc}')
    logging.info(f'Score = AUC - MAE: {auc - mae}')


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("./config.ini")

    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.FileHandler(f'{config["REPORT"]["SaveDir"]}report.log'),
            logging.StreamHandler(sys.stdout),
        ],
    )
    main(cfg=config)
