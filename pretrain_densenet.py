from pretrainer import Pretrainer
from res_densenet import OurDenseNet
from util import format_filename


epochs = 20
batch_size = 100
composers = ['Scarlatti', 'Rachmaninov', 'Grieg', 'Buxehude', 'Debussy', 'Albéniz', 'Schumann', 'German', 'Skriabin',
             'Tchaikovsky', 'Chaminade', 'Burgmuller', 'Paganini', 'Hummel', 'Czerny', 'Joplin', 'Liszt', 'Dvorak']

file_name = format_filename("densenet_test", ("precision8", epochs, "adadelta"), add_date=False)

pretrainer = Pretrainer(
    model_class=OurDenseNet,
    file_name=file_name,
    composers=composers,
    num_classes=len(composers),
    epochs=epochs,
    batch_size=batch_size,
    pretrained=False,
    verbose=False
)

pretrainer.pretrain()
