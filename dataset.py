import numpy as np
from os import listdir
from os.path import join, isdir, isfile
from places_model import Places_CNN
import h5py
from tqdm import tqdm

# Set dataset path
path_15scene = '/home/nick/local/datasets/15_scene/scene_categories'
# The preprocessed dataset will be saved in the following file
h5_path = '15scene_feats.h5'

def load_15_scene_dataset(path=path_15scene):
    categories = [f for f in listdir(path) if isdir(join(path, f))]
    files = []
    labels = []
    i = 0
    for cat in categories:
        c_files = [join(join(path, cat), f) for f in listdir(join(path, cat))]
        files.extend(c_files)
        labels.extend([i for x in c_files])
        i += 1

    return files, labels, categories


def get_15scene_features(h5_path=h5_path):
    data_paths, labels, cats = load_15_scene_dataset()

    if not isfile(h5_path):
        print "Extracting features..."
        cnn = FeatureExractor()
        cnn.extract_features(h5_path, data_paths)

    f = h5py.File(h5_path, "r")
    return f['features'], np.asarray(labels, dtype='int32')


def get_15scene_splits(n_train=100, seed=1, path=path_15scene):
    files, labels, cats = load_15_scene_dataset(path)
    np.random.seed(seed)
    labels = np.asarray(labels)

    # Get splits
    train_idx = []
    test_idx = []

    for i in range(15):
        idx = (labels == i)
        idx = np.where(idx)[0]
        np.random.shuffle(idx)
        train_idx.extend(idx[:n_train])
        test_idx.extend(idx[n_train:])

    train_idx = np.asarray(train_idx)
    test_idx = np.asarray(test_idx)

    return train_idx, test_idx


class FeatureExractor:
    def __init__(self):
        self.cnn = Places_CNN()

    def extract_features(self, filename, data_paths):
        f = h5py.File(filename, "w")
        img_features = f.create_dataset("features", (len(data_paths), 14 * 14, 512), dtype='float16')

        for i in tqdm(range(len(data_paths))):
            im = self.cnn.load_image(data_paths[i])
            self.cnn.process_batch([im])
            features = self.cnn.get_conv_features().reshape(-1, 14 * 14, 512)
            img_features[i] = features
