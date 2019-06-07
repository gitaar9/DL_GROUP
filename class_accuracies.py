import argparse

from torch.utils.data import DataLoader

from datasets import MidiClassicMusic, Mode
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
        test_loader = DataLoader(
            MidiClassicMusic(folder_path="./data/midi_files_npy_8_40", mode=Mode.TEST, slices=40,
                             composers=[composer],
                             cv_cycle=0),
            batch_size=100,
            shuffle=False
        )
        print("size of testset: {}".format(len(test_loader)))
        for network_name in ["best_models/{}_run{}".format(filename, file_number) for file_number in range(amount_of_files)]:
            net.load_model(network_name)
            _, _, _, _, acc_list = net.validate(test_loader)
            print(acc_list)
            acc += sum(acc_list)/len(test_loader)
        composers_accuracies.append(acc / 4)

    for name, accuracy in zip(composers, composers_accuracies):
        print("{}: {}".format(name, accuracy))
