class MetricsComputer:
    def __init__(self, module):
        self.module = module
        self.rc = self.module.rc

    def should_compute(self, stage):
        return (
            (stage == 'train' and self.rc.config.save_train_metrics)
            or (stage == 'val' and self.rc.config.save_val_metrics)
            or (stage == 'test' and self.rc.config.save_test_metrics)
        )

    def compute(self, stage=None, monitor_value=None):
        if not self.should_compute(stage):
            if not self.module.hparams.interleaved or self.module.current_epoch % 2 == 1:
                # If the base loss is computed during this epoch
                return {
                    self.module.monitor: monitor_value,
                }
            return {}

        if stage == 'train':
            return self.train_metrics()
        elif stage == 'val':
            return self.val_metrics()
        elif stage == 'test':
            return self.test_metrics()
        else:
            raise RuntimeError('Invalid stage')
