Accompanying repository for the paper *A Large-Scale Study of Probabilistic Calibration in Neural Network Regression*. ICML, 2023.

### Overview

This repository includes the implementation of:
- Various calibration methods (quantile recalibration, conformalized quantile prediction, quantile regularization, PCE-KDE, PCE-Sort...) with hyperparameter tuning.
- Various metrics (NLL, CRPS, quantile score, probabilistic calibration error...).
- The pipeline to download a total of 57 tabular regression datasets from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/) and [OpenML](https://www.openml.org/).
- Various plots (reliability diagrams, boxen plots, CD diagrams...).

<p align="center">
<img src="images/pce_and_rel_diags_mix_nll.svg?raw=true" alt="" width="88%" align="top">
</p>

<p align="center">
<img src="images/calib_l1.svg?raw=true" alt="" width="44%" align="top">
<img src="images/crps.svg?raw=true" alt="" width="44%" align="top">
</p>

### Abstract

Accurate probabilistic predictions are essential for optimal decision making. While neural network miscalibration has been studied primarily in classification, we investigate this in the less-explored domain of regression. We conduct the largest empirical study to date to assess the probabilistic calibration of neural networks. We also analyze the performance of recalibration, conformal, and regularization methods to enhance probabilistic calibration. Additionally, we introduce novel differentiable recalibration and regularization methods, uncovering new insights into their effectiveness. Our findings reveal that regularization methods offer a favorable tradeoff between calibration and sharpness. Post-hoc methods exhibit superior probabilistic calibration, which we attribute to the finite-sample coverage guarantee of conformal prediction. Furthermore, we demonstrate that quantile recalibration can be considered as a specific case of conformal prediction. Our study is fully reproducible and implemented in a common code base for fair comparisons.

### Installation

The experiments have been run in python 3.9 with the package versions listed in `requirements.txt`.

They can be installed using:
```
pip install -r requirements.txt
```

### Main experiments

The main experiments can be run using:
```
python run.py name="full" nb_workers=1 repeat_tuning=5 \
        log_base_dir="logs" progress_bar=False \
        save_train_metrics=False save_val_metrics=False remove_checkpoints=True \
        selected_dataset_groups=["uci","oml_297","oml_299","oml_269"] \
        tuning_type="regul_and_posthoc"
```

Note that interrupting and running the same command again will skip the models that have already been computed. The parameter `nb_workers` allows to run multiple experiments at the same time.

Then, the corresponding figures can be created in the notebook `main_figures.ipynb`.

### Experiments related to tuning

Experiments related to tuning can be run using:
```
python run.py name="hparam_tuning" nb_workers=1 repeat_tuning=5 \
        log_base_dir="logs" progress_bar=False \
        save_train_metrics=False save_val_metrics=False remove_checkpoints=True \
        selected_dataset_groups=["uci","oml_297","oml_299","oml_269"] \
        tuning_type="hparam_tuning"
```

Then, the corresponding figures can be created in the notebook `hparam_figures.ipynb`.

### Cite

Please cite our paper if you use this code in your own work:
```
@inproceedings{DheurICML2023,
  title     = {A Large-Scale Study of Probabilistic Calibration in Neural Network Regression},
  author    = {Dheur, Victor and Ben taieb, Souhaib},
  booktitle = {The 40th International Conference on Machine Learning},
  year      = {2023},
}
```
