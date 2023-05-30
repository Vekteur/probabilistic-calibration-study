class DefaultTrainer:
    def __init__(self, **kwargs):
        pass

    def fit(self, model, datamodule):
        self.datamodule = datamodule
        model.trainer = self
        model.fit(datamodule)

    def test(self, model, datamodule, **kwargs):
        model.test(datamodule)
