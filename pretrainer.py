class Pretrainer:

    def __init__(self, model_class, file_name, **model_kwargs):
        """
        :param model_class: A class extending from Basenet
        :param file_name: The base-filename of the saved results
        :param kwargs: The arguments that should be passed to model_classes constructor
        """
        self.model_class = model_class
        self.file_name = file_name
        self.model_kwargs = model_kwargs

    def pretrain(self):
        net = None
        for fold in range(4):
            print("{}\nFold {} for {}\n{}".format('#' * 30, fold, self.file_name, '#' * 30))

            if net is None:
                net = self.model_class(cv_cycle=fold, **self.model_kwargs)
            else:
                net.change_data_loaders(self.model_kwargs['batch_size'], fold)

            metrics = net.run()
            filename = "pretrain_results/{}_run{}".format(self.file_name, fold)
            net.save_metrics(filename, metrics)

        net.save_model('pretrained_models/{}'.format(self.file_name))
