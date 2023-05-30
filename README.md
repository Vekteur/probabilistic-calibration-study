Accompanying repository for the paper *A Large-Scale Study of Probabilistic Calibration in Neural Network Regression*. ICML, 2023.

### Abstract

Accurate estimation of predictive uncertainty is essential for optimal decision making.
However, recent works have shown that current neural networks tend to be miscalibrated, sparking interest in different approaches to calibration.
In this paper, we conduct a large-scale empirical study of the probabilistic calibration of neural networks on 57 tabular regression datasets.
We consider recalibration, conformal and regularization approaches, and investigate the trade-offs they induce on calibration and sharpness of the predictions.
Based on kernel density estimation, we design new differentiable recalibration and regularization methods, yielding new insights into the performance of these approaches.
Furthermore, we find conditions under which recalibration and conformal prediction are equivalent.
Our study is fully reproducible and implemented in a common code base for fair comparison.

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
Then, the corresponding figures can be created in the notebook `main_figures.ipynb`.

### Experiments related to tuning

Experiments related to tuning can be run using:
```
python -u run.py name="hparam_tuning" nb_workers=1 repeat_tuning=5 \
        log_base_dir="logs" progress_bar=False \
        save_train_metrics=False save_val_metrics=False remove_checkpoints=True \
        selected_dataset_groups=["uci","oml_297","oml_299","oml_269"] \
        tuning_type="hparam_tuning"
```
Then, the corresponding figures can be created in the notebook `hparam_figures.ipynb`.
