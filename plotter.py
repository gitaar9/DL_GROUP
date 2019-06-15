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
    all_data = []
    for file_path in ["{}_run{}".format(filename, file_number) for file_number in range(amount_of_files)]:
        try:
            all_data.append(read_in_file(file_path))
        except FileNotFoundError:
            print("\nWARNING: {} is missing so not plotting all folds.\n".format(file_path))
    all_data = np.array(all_data)

    averages = np.average(all_data, axis=0)
    averages[:, 5:8] *= 100  # Back to percentages

    stds = np.std(all_data, axis=0)
    stds[:, 5:8] *= 100  # Back to percentages

    return averages, stds


def plot_collumns(list_of_metrics, collumns, y_label, legend_names, colors=None):
    colors = colors or plt.rcParams['axes.prop_cycle'].by_key()['color']
    for idx, metrics in enumerate(list_of_metrics):
        for collumn in collumns:
            plt.plot(metrics[:, collumn], colors[idx], label=legend_names[idx])
    plt.ylabel(y_label)
    plt.xlabel('Epochs')
    plt.xlim([-1, max([m.shape[0] for m in list_of_metrics])])
    plt.legend(loc='lower right')
    plt.show()


def plot_multiple_accuracies(list_of_averages, legend_names, colors=None):
    colors = colors or plt.rcParams['axes.prop_cycle'].by_key()['color']
    for idx, averages in enumerate(list_of_averages):
        plt.plot(averages[:, 5], colors[idx], label=legend_names[idx])
        if averages.shape[1] > 6:
            plt.plot(averages[:, 6], colors[idx], linestyle='--')
            # plt.plot(averages[:, 7], colors[idx], linestyle=':')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Epochs')
    plt.xlim([-1, max([avg.shape[0] for avg in list_of_averages])])
    plt.legend(loc='lower right')
    plt.show()


def plot_multiple_loss(list_of_averages, legend_names, colors=None):
    colors = colors or plt.rcParams['axes.prop_cycle'].by_key()['color']
    for idx, averages in enumerate(list_of_averages):
        plt.plot(averages[:, 0], color=colors[idx], linestyle='--') #, label=legend_names[idx]+' train loss')
        plt.plot(averages[:, 1], color=colors[idx], label=legend_names[idx])
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.xlim([-1, max([avg.shape[0] for avg in list_of_averages])])
    plt.legend(loc='upper right')
    plt.show()


def print_summary(filenames, amount_of_files, legend_names=None):
    legend_names = legend_names or map(lambda f: f.replace('advanced_lstm', 'a_lstm'), filenames)
    averages_and_stds = [read_in_files_to_average(filename, amount_of_files) for filename in filenames]

    # Print final accuracies + stds
    tabs = 7
    print("\t" * tabs + "training_loss\tvalidation_loss\tprecision\trecall\t\tf1\t\taccuracy\ttest_accuracy\tbest")
    for filename, (averages, stds) in zip(legend_names, averages_and_stds):
        s = filename.split('/')[-1]
        s += "\t" * (tabs - int(len(s) / 8))
        for average, std in zip(averages[-1, :], stds[-1, :]):
            s += "%0.2f+/-%0.2f\t" % (average, std)
        print(s)


def plot_selecting(filenames, amount_of_files=1, legend_names=None):
    # Read in the data per file or average over amount_of_files files:
    if amount_of_files > 1:
        all_data = [read_in_files_to_average(each_file, amount_of_files)[0] for each_file in filenames]
    else:
        all_data = [read_in_file(f) for f in filenames]

    # Use legend_names if provided otherwise use the filenames as legend names
    legend_names = legend_names or filenames
    legend_names = list(map(lambda n: n.replace('_', '-'), legend_names))

    # Plot the accuracies and losses
    plot_multiple_accuracies(all_data, legend_names)
    plot_multiple_loss(all_data, legend_names)
    #plot_collumns(all_data, [4], 'F1 score', legend_names)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Averages and plots output fom our pytorch networks.')
    parser.add_argument('filenames', type=str, nargs='+',
                        help='The base filename of the files we want to read in when using amount_of_files > 1.'
                        'Otherwise just full filenames of the files.')
    parser.add_argument('--amount_of_files', type=int, default=1, help='The amount of runs we did.')
    parser.add_argument('--legend', type=str, default=[], nargs='+', help='The legend names of the filenames.')
    parser.add_argument('--print_summary', default=False, action='store_true',
                        help='When amount_of_files > 1 this can be used to print mean and stds of the results.')

    args = parser.parse_args()
    return args.amount_of_files, args.filenames, args.legend, args.print_summary


if __name__ == '__main__':
    amount_of_files, filenames, legend_names, summary = parse_arguments()

    if summary:
        print_summary(filenames, amount_of_files, legend_names)

    plot_selecting(filenames, amount_of_files, legend_names)
