import argparse

from res_densenet import OurDenseNet


def parse_arguments():
    parser = argparse.ArgumentParser(description='FILL IN LATER')
    parser.add_argument('filename', type=str,
                        help='The base filename of the files we want to read in when using amount_of_files > 1.'
                        'Otherwise just full filenames of the files.')
    parser.add_argument('--amount_of_files', type=int, default=1, help='The amount of runs we did.')

    args = parser.parse_args()
    return args.amount_of_files, args.filename


if __name__ == '__main__':
    amount_of_files, filename = parse_arguments()

    composers = ['Brahms', 'Mozart', 'Schubert', 'Mendelsonn', 'Haydn', 'Vivaldi', 'Clementi', 'Beethoven', 'Haendel',
                 'Bach', 'Chopin']

    net = OurDenseNet(epochs=1, save_path="-", num_classes=11, batch_size=100, composers=['Brahms'])

    composers_accuracies = []
    for composer in composers:
        print("Getting accuracy for {}".format(composer))
        acc = 0
        net.change_data_loaders(batch_size=100, cv_cycle=0, composers=[composer])
        for network_name in ["best_models/{}_run{}".format(filename, file_number) for file_number in range(amount_of_files)]:
            net.load_model(network_name)
            _, _, _, _, acc_list = net.validate(net.test_loader)
            print(acc_list)
            acc += sum(acc_list)/len(net.test_loader)
        composers_accuracies.append(acc / 4)

    for name, accuracy in zip(composers, composers_accuracies):
        print("{}: {}".format(name, accuracy))
