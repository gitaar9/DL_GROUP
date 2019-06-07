class CrossValidator:

    def __init__(self, model_class, file_name, run_type, **model_kwargs):
        """
        :param model_class: A class extending from Basenet
        :param run_type: The way the classifier will be trained  composers/era/country/year
        :param file_name: The base-filename of the saved results
        :param kwargs: The arguments that should be passed to model_classes constructor
        """
        self.model_class = model_class
        self.run_type = run_type
        self.file_name = file_name
        self.model_kwargs = model_kwargs

    def cross_validate(self):
        for fold in range(4):
            print("{}\nFold {} for {}\n{}".format('#' * 30, fold, self.file_name, '#' * 30))
            net = self.model_class(cv_cycle=fold, run_type=self.run_type, **self.model_kwargs)
            metrics = net.run()
            filename = "cross_validated_results/{}_run{}".format(self.file_name, fold)
            net.save_metrics(filename, metrics)
            net.free()

    def single_run(self):
        print("{}\nSingle run of {}\n{}".format('#' * 30, self.file_name, '#' * 30))
        net = self.model_class(run_type=self.run_type, **self.model_kwargs)
        metrics = net.run()
        filename = "results/{}_single_run".format(self.file_name)
        net.save_metrics(filename, metrics)
        net.free()
