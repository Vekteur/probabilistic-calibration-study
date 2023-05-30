from collections import defaultdict

import torch


class MetricsCollector:
    def __init__(self, module):
        self.metrics = defaultdict(lambda: defaultdict(dict))
        self.module = module

    def collect_per_step(self, outputs, stage):
        average_metrics = defaultdict(list)
        for metrics_at_this_epoch in outputs:
            for key, value in metrics_at_this_epoch.items():
                average_metrics[key].append(value)
        for key, values in average_metrics.items():
            self.metrics['per_epoch'][f'{stage}_{key}'][self.module.current_epoch] = (
                torch.stack(values).mean(dim=0).item()
            )

    def collect_per_epoch(self, metrics, stage):
        for key, value in metrics.items():
            self.metrics['per_epoch'][f'{stage}_{key}'][self.module.current_epoch] = value.item()

    def get_best_score_and_iter(self):
        best_score = float('inf')
        best_iter = -1
        for iter, score in self.metrics['per_epoch'][f'val_{self.module.monitor}'].items():
            if score < best_score:
                best_score = score
                best_iter = iter
        return best_score, best_iter

    def add_best_iter_metrics(self):
        (
            self.metrics['best_score'],
            self.metrics['best_iter'],
        ) = self.get_best_score_and_iter()

    def advance_timer(self, timer, amount):
        if timer not in self.metrics:
            self.metrics[timer] = 0
        self.metrics[timer] += amount
