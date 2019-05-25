

class CrossValidator:

    def __init__(self, model_class, file_name, **kwargs):
        self.model_class = model_class
        self.file_name = file_name
        self.model_kwargs = kwargs

    def cross_validate(self):
        for fold in range(4):
            net = self.model_class(cv_cycle=fold, **self.model_kwargs)
            metrics = net.run()
            filename = "results/{}_run{}".format(self.file_name, fold)
            net.save_metrics(filename, metrics)
