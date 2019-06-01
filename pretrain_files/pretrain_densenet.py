from res_densenet import OurDenseNet
from util import format_filename


if __name__ == '__main__':
    epochs = 10
    batch_size = 100
    composers = ['Scarlatti', 'Rachmaninov', 'Grieg', 'Buxehude', 'Debussy', 'Albéniz', 'Schumann', 'German', 'Skriabin',
                 'Tchaikovsky', 'Chaminade', 'Burgmuller', 'Paganini', 'Hummel', 'Czerny', 'Joplin', 'Liszt', 'Dvorak']

    file_name = format_filename("densenet_test", ("precision8", epochs, "adadelta"))

    net = None
    for fold in range(4):
        if net is None:
            net = OurDenseNet(cv_cycle=fold, composers=composers, num_classes=len(composers), epochs=epochs,
                              batch_size=batch_size, verbose=False)
        else:
            net.change_data_loaders(batch_size, fold)

        print("{}\nFold {} for {}\n{}".format('#' * 30, fold, file_name, '#' * 30))
        metrics = net.run()
        filename = "pretrain_results/{}_run{}".format(file_name, fold)
        net.save_metrics(filename, metrics)
    net.save_model('pretrained_models/{}'.format(file_name))
