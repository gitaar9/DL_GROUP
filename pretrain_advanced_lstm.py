import argparse

from advanced_lstm import OurLSTM
from pretrainer import Pretrainer
from util import format_filename


def parse_arguments():
    parser = argparse.ArgumentParser(description='Pretrain some lstms on the midi database.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='The amount of epochs that the model will be trained.')
    parser.add_argument('--num_layers', type=int, default=1,
                        help='The lstm layers.')
    parser.add_argument('--hidden_size', type=int, default=8,
                        help='The amount of blocks in every lstm layer.')
    parser.add_argument('--dropout', type=float, default=.5,
                        help='The dropout rate after each lstm layer.')
    parser.add_argument('--chunks', type=int, default=10,
                        help='How many chunks to make of the input sequence.')
    parser.add_argument('--optimizer', type=str, default='Adadelta',
                        help='Please decide which optimizer you want to use: Adam or Adadelta')

    args = parser.parse_args()

    return args.epochs, args.optimizer, args.num_layers, args.hidden_size, args.dropout, args.chunks


if __name__ == '__main__':
    arguments = parse_arguments()

    composers = ['Scarlatti', 'Rachmaninov', 'Grieg', 'Buxehude', 'Debussy', 'Albéniz', 'Schumann', 'German', 'Skriabin',
                 'Tchaikovsky', 'Chaminade', 'Burgmuller', 'Paganini', 'Hummel', 'Czerny', 'Joplin', 'Liszt', 'Dvorak']

    file_name = format_filename("advanced_lstm_test", ("precision8", ) + arguments, add_date=False)

    # Unpack the commandline arguments for use
    epochs, optimizer, num_layers, hidden_size, dropout, chunks = arguments

    pretrainer = Pretrainer(
        model_class=OurLSTM,
        file_name=file_name,
        composers=composers,
        num_classes=len(composers),
        epochs=epochs,
        batch_size=100,
        num_layers=num_layers,
        hidden_size=hidden_size,
        dropout=dropout,
        n_chunks=chunks,
        optimizer=optimizer,
        verbose=False
    )

    pretrainer.pretrain()
