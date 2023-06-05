Accompanying repository for the paper *A Large-Scale Study of Probabilistic Calibration in Neural Network Regression*. ICML, 2023.

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
