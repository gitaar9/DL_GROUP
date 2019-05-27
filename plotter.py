import argparse
import numpy as np
import csv
import matplotlib.pyplot as plt


def read_in_file(filename):
    with open(filename, 'r') as f:
        data = np.array(list(csv.reader(f))[1:]).astype(float)

    if data.shape[1] == 7:
        # If there is test accuracies, select the idx of the best validation accuracy and
        # add the test accuracy with that index as a line to be plotted
        test_accuracy = data[np.argmax(data[:, 5]), 6]
        data = np.concatenate((data, np.ones((data.shape[0], 1)) * test_accuracy), axis=1)

    return data


def read_in_files_to_average(filename, amount_of_files):
    all_data = [read_in_file("{}_run{}".format(filename, file_number)) for file_number in range(amount_of_files)]

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
        plt.plot(averages[:, 5], colors[idx], label=legend_names[idx])
        if averages.shape[1] > 6:
            # plt.plot(averages[:, 6], colors[idx], linestyle='--')
            plt.plot(averages[:, 7], colors[idx], linestyle=':')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Epochs')
    plt.xlim([-1, averages.shape[0]])
    plt.legend(loc='lower right')
    plt.show()


def plot_multiple_loss(list_of_averages, legend_names, colors=None):
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] if colors is None else colors
    for idx, averages in enumerate(list_of_averages):
        plt.plot(averages[:, 0], color=colors[idx], linestyle='--') #, label=legend_names[idx]+' train loss')
        plt.plot(averages[:, 1], color=colors[idx], label=legend_names[idx])
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.xlim([-1, averages.shape[0]])
    plt.legend(loc='upper right')
    plt.show()


def print_summary(filenames, amount_of_files):
    # Print final accuracies + stds
    print("\t\t\t\ttraining_loss\tvalidation_loss\tprecision\trecall\t\tf1\t\taccuracy\ttest_accuracy\tbest")
    averages_and_stds = [read_in_files_to_average(filename, amount_of_files) for filename in filenames]
    for filename, (averages, stds) in zip(filenames, averages_and_stds):
        s = filename.split('/')[-1] + "\t" + ("\t" if "adam" in filename and "res_" in filename else "")
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


def parse_arguments():
    parser = argparse.ArgumentParser(description='Averages and plots output fom our pytorch networks.')
    parser.add_argument('--amount_of_files', type=int, default=1,
                        help='The amount of runs we did.')
    parser.add_argument('--filenames', type=str, default=[], nargs='+',
                        help='The base filename of the files we want to read in.')
    parser.add_argument('--legend', type=str, default=[], nargs='+',
                        help='The legend names of the filenames.')
    parser.add_argument('--print_summary', default=False, action='store_true',
                        help='Just plot all final results for the Deep Learning assignment')

    args = parser.parse_args()
    return args.amount_of_files, args.filenames, args.legend, args.print_summary


if __name__ == '__main__':
    amount_of_files, filenames, legend_names, summary = parse_arguments()

    plot_selecting(filenames, amount_of_files, legend_names)

    if summary:
        print_summary(filenames, amount_of_files)

