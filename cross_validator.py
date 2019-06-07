class CrossValidator:

    def __init__(self, model_class, file_name, folder="cross_validated_results", **model_kwargs):
        """
        :param model_class: A class extending from Basenet
        :param file_name: The base-filename of the saved results
        :param kwargs: The arguments that should be passed to model_classes constructor
        """
        self.model_class = model_class
        self.file_name = file_name
        self.model_kwargs = model_kwargs
        self.folder = folder

    def cross_validate(self):
        for fold in range(4):
            print("{}\nFold {} for {}\n{}".format('#' * 30, fold, self.file_name, '#' * 30))
            save_path = "best_models/{}_run{}".format(self.file_name, fold)
            net = self.model_class(cv_cycle=fold, save_path=save_path, **self.model_kwargs)
            metrics = net.run()
            filename = "{}/{}_run{}".format(self.folder, self.file_name, fold)
            net.save_metrics(filename, metrics)
            net.free()

    def single_run(self):
        print("{}\nSingle run of {}\n{}".format('#' * 30, self.file_name, '#' * 30))
        save_path = "best_models/{}_single_run".format(self.file_name)
        net = self.model_class(save_path=save_path, **self.model_kwargs)
        metrics = net.run()
        filename = "results/{}_single_run".format(self.file_name)
        net.save_metrics(filename, metrics)
        net.free()
