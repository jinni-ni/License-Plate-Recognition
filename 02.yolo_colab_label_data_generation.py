import os

current_path = os.path.abspath(os.curdir)

COLAB_DARKNET_ESCAPE_PATH = '/content/gdrive/My\ Drive/darknet'
COLAB_DARKNET_PATH = '/content/gdrive/My Drive/darknet'

YOLO_IMAGE_PATH = current_path + '/blackbox-data/train/images'
YOLO_FORMAT_PATH = current_path + '/blackbox-data/train/images'

class_count = 0
test_percentage = 0.2
paths = []

# create names file
with open(YOLO_FORMAT_PATH + '/' + 'classes.names', 'w') as names, \
     open(YOLO_FORMAT_PATH + '/' + 'classes.txt', 'r') as txt:
    for line in txt:
        names.write(line)
        class_count += 1
    print("[classes.names] is created")


# create custom_dasta.data
with open(YOLO_FORMAT_PATH + '/' 'custom_data.data', 'w') as data:
    data.write('classes = ' + str(class_count) + '\n')
    data.write('train = ' + COLAB_DARKNET_ESCAPE_PATH + '/blackbox-data/train/images/' + 'train.txt' + '\n')
    data.write('valid = ' + COLAB_DARKNET_ESCAPE_PATH + '/blackbox-data/train/images/' + 'test.txt' + '\n')
    data.write('names = ' + COLAB_DARKNET_ESCAPE_PATH + '/blackbox-data/train/images/' + 'classes.names' + '\n')
    data.write('backup = backup')
    print("[custom_data.data] is created")

# custom dir
os.chdir(YOLO_IMAGE_PATH)
for current_dir, dirs, files in os.walk('.'):
    for f in files:
        if f.endswith('.png'):
            image_path = COLAB_DARKNET_PATH + '/blackbox-data/train/images/' + f
            paths.append(image_path + '\n')

paths_test = paths[:int(len(paths) * test_percentage)]
paths = paths[int(len(paths) * test_percentage):]

with open(YOLO_FORMAT_PATH + '/' + 'train.txt', 'w') as train_txt:
    for path in paths:
        train_txt.write(path)
    print("[train.txt] is created")

with open(YOLO_FORMAT_PATH + '/' + 'test.txt', 'w') as test_txt:
    for path in paths_test:
        test_txt.write(path)
    print("[test.txt] is created")
