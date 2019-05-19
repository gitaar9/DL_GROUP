import argparse
import numpy as np
import csv
import matplotlib.pyplot as plt


def parse_arguments():
    parser = argparse.ArgumentParser(description='Averages and plots output fom our pytorch networks.')
    parser.add_argument('--amount_of_files', type=int, default=1,
                        help='The amount of runs we did.')
    parser.add_argument('--filename', type=str, default=[], nargs='+',
                        help='The base filename of the files we want to read in.')
    parser.add_argument('--legend', type=str, default=[], nargs='+',
                        help='The legend names of the filenames.')
    parser.add_argument('--everything', default=False, action='store_true',
                        help='Just plot all final results for the Deep Learning assignment')

    args = parser.parse_args()
    return args.amount_of_files, args.filename, args.everything, args.legend


def read_in_file(filename):
    with open(filename, 'r') as f:
        data = np.array(list(csv.reader(f))[1:]).astype(float)
    return data


def read_in_files_to_average(filename, amount_of_files):
    all_data = [read_in_file("results/{}_{}".format(filename, file_number)) for file_number in range(amount_of_files)]
    until_epoch = 400
    all_data = list(map(lambda r: r[:until_epoch], all_data))

    # u = E(x) / n
    averages = sum(all_data) / amount_of_files

    # std = sqrt( E((x - u)^2) / n)
    stds = np.sqrt(sum(map(lambda d: np.square(d-averages), all_data)) / amount_of_files)

    return averages, stds


def plot_accuracy(averages, filename):
    plt.plot(averages[:, 5], label=filename)
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(loc='bottom right')
    plt.show()


def plot_losses(averages, filename):
    plt.plot(averages[:, 0], label='test_loss'+filename)
    plt.plot(averages[:, 1], label='validation_loss'+filename)
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(loc='bottom right')
    plt.show()


def plot_multiple_accuracies(list_of_averages, legend_names, colors=None):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] if colors is None else colors
    for idx, averages in enumerate(list_of_averages):
        plt.plot(averages[:, 5], colors[idx])
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Epochs')
    # plt.axis([-1, 400, .65, 1])
    plt.legend(legend_names, loc='lower right')
    plt.show()


def plot_multiple_loss(list_of_averages, legend_names, colors=None):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] if colors is None else colors
    for idx, averages in enumerate(list_of_averages):
        plt.plot(averages[:, 0], color=colors[idx], linestyle='--', label=legend_names[idx]+' test loss')
        plt.plot(averages[:, 1], color=colors[idx], label=legend_names[idx]+' validation loss')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    # plt.axis([-1, 400, 0, 1.5])
    plt.legend(loc='upper right')
    plt.show()


def print_everything(filenames):
    # Print final accuracies + stds
    print("\t\t\ttraining_loss\tvalidation_loss\tprecision\trecall\t\tf1\t\taccuracy")
    averages_and_stds = [read_in_files_to_average(filename, 4) for filename in filenames]
    for filename, (averages, stds) in zip(filenames, averages_and_stds):
        s = filename + "\t" + ("\t" if "adam" in filename and "res_" in filename else "")
        for average, std in zip(averages[-1, :], stds[-1, :]):
            s += "%0.3f+/-%0.3f\t" % (average, std)
        print(s)


def plot_selecting(filenames, amount_of_files, legend_names):
    if amount_of_files > 1:
        all_data = [read_in_files_to_average(each_file, amount_of_files)[0] for each_file in filenames]
    else:
        all_data = [read_in_file(f) for f in filenames]
    legend_names = list(map(lambda n: n.replace('_', '-'), legend_names))
    plot_multiple_accuracies(all_data, legend_names if legend_names else filenames)
    plot_multiple_loss(all_data, legend_names if legend_names else filenames)


if __name__ == '__main__':
    amount_of_files, filename, everything, legend_names = parse_arguments()

    plot_selecting(filename, amount_of_files, legend_names)


# python plotter.py --filename results/lstm_test2_100_2_8 results/lstm_test2_100_2_16 results/lstm_test2_100_2_32 results/lstm_test2_100_2_64
# python plotter.py --filename results/lstm_test2_100_1_8 results/lstm_test2_100_1_16 results/lstm_test2_100_1_32 results/lstm_test2_100_1_64