from cross_validator import CrossValidator
from simple_dense import OurSimpleDense
from datasets import MidiClassicMusic


composers = ['Brahms', 'Mozart', 'Schubert', 'Mendelsonn', 'Haydn', 'Beethoven', 'Bach', 'Chopin']

# run_type specifies the type of classification, default should be 'composers'
# other options are 'era' or 'country' as year does not work yet
run_type = 'era'
#run_type = 'year'
midi = MidiClassicMusic(composers=composers, run_type=run_type)
num_classes = len(midi.classes)
print(midi.classes)
#num_classes = 1

dropout = 0.6
epochs = 30
block_config = [2,2]
block_config_string = '(' + ','.join([str(i) for i in block_config]) + ')'
file_name = "dense_test_precision8_{}_{}_{}".format(epochs, dropout, block_config_string)

# make sure to copy run_type through cross_validator -> networks -> data loaders -> midiclassicmusic
cv = CrossValidator(
    model_class=OurSimpleDense,
    file_name=file_name,
    composers=composers,
    run_type=run_type,
    num_classes=num_classes,
    epochs=epochs,
    batch_size=100,
    verbose=False,
    dropout=dropout,
    block_config=block_config
)

cv.cross_validate()