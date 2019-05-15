import argparse
import numpy as np
import csv
import matplotlib.pyplot as plt


def parse_arguments():
    parser = argparse.ArgumentParser(description='Averages and plots output fom our pytorch networks.')
    parser.add_argument('--amount_of_files', type=int, default=4,
                        help='The amount of runs we did.')
    parser.add_argument('--filename', type=str, default='default',
                        help='The base filename of the files we want to read in.')
    parser.add_argument('--plot_lstm_test_100', default=False, action='store_true')

    args = parser.parse_args()
    return args.amount_of_files, args.filename, args.plot_lstm_test_100


def read_in_file(filename):
    with open(filename, 'r') as f:
        data = np.array(list(csv.reader(f))[1:]).astype(float)
    return data


def read_in_files_to_average(filename, amount_of_files):
    all_data = [read_in_file("{}_{}".format(filename, file_number)) for file_number in range(amount_of_files)]

    # u = E(x) / n
    averages = sum(all_data) / amount_of_files

    # std = sqrt( E((x - u)^2) / n)
    stds = np.sqrt(sum(map(lambda d: np.square(d-averages), all_data)) / amount_of_files)

    return averages, stds


def plot_collumn(data, collumn=5, ylabel='Accuracy'):
    plt.plot(data[:, collumn])
    plt.ylabel(ylabel)
    plt.xlabel('Epochs')
    plt.show()


def plot_multiple_file_data_arrays(list_of_averages, filenames, collumn=5, ylabel='Accuracy', loc='lower right'):
    for averages in list_of_averages:
        plt.plot(averages[:, collumn])
    plt.ylabel(ylabel)
    plt.xlabel('Epochs')
    plt.legend(filenames, loc=loc)
    plt.show()


def plot_lstm_test_100():
    layer_options = ['1', '2']
    hidden_size_options = ['8', '16', '32', '64']
    filenames = ["results/lstm_test_100_{}_{}".format(l, h) for l in layer_options for h in hidden_size_options]
    all_averages = [read_in_file(filename) for filename in filenames]
    plot_multiple_file_data_arrays(all_averages, filenames, 5, 'Accuracy')
    plot_multiple_file_data_arrays(all_averages, filenames, 0, 'Training loss', 'upper right')
    plot_multiple_file_data_arrays(all_averages, filenames, 1, 'Validation loss', 'upper left')


def print_accuracies_and_stds(filenames):
    # Print final accuracies + stds
    print("\t\t\t\ttraining_loss\tvalidation_loss\tprecision\trecall\t\tf1\t\taccuracy")
    averages_and_stds = [read_in_files_to_average(filename, 4) for filename in filenames]
    for filename, (averages, stds) in zip(filenames, averages_and_stds):
        s = filename + "\t" + ("\t" if "adam" in filename and "res_" in filename else "")
        for average, std in zip(averages[-1,:], stds[-1,:]):
            s += "%0.3f+/-%0.3f\t" % (average, std)
        print(s)


if __name__ == '__main__':
    amount_of_files, filename, plot_lstm_test = parse_arguments()

    if plot_lstm_test:
        plot_lstm_test_100()
    else:
        averages = read_in_files_to_average(filename, amount_of_files)[0]
        plot_collumn(averages, 5, 'Accuracy')
        plot_collumn(averages, 0, 'Training loss')
        plot_collumn(averages, 1, 'Validation loss')
