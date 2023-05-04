import os
import re
import nibabel as nib
import numpy as np
import cv2
import data_augmentation as da
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split

# Define the directories where the scans are located
TRAIN_DATASET_PATH = '/content/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/'
VALIDATION_DATASET_PATH = '/content/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData'


train_dirs = [f.path for f in os.scandir(TRAIN_DATASET_PATH) if f.is_dir()]
validation_dirs = [f.path for f in os.scandir(
    VALIDATION_DATASET_PATH) if f.is_dir()]

# Extract the id of the volume from the directory name


def get_id_from_path(path):
    m = re.search('BraTS20_(Training|Validation)_(\d+)', path)
    return m.group(2)

# Remove all volumes that don't contain all necessary files specified in look_for


def _get_ids_to_remove(directories, look_for):
    delete_ids = []
    for directory in directories:
        for _, __, files in os.walk(directory):
            delete = False
            for word in look_for:
                if not any(word in x for x in files):
                    delete = True
                    break
            if delete:

                id = get_id_from_path(directory)
                delete_ids.append(id)
    return delete_ids


# Get the ids of the volumes that contain all necessary files
def _get_ids_completed(dirs, mri_types, train=False):
    # Make a copy of mri types for look_for
    look_for = mri_types[:]
    # Train set should contain seg files
    if train and not 'seg' in mri_types:
        look_for += ['seg']
    # remove volume if doesn't contain all necessary files
    delete_ids = _get_ids_to_remove(dirs, look_for)
    ids = list(set([get_id_from_path(p) for p in dirs]))
    for id in delete_ids:
        ids.remove(id)

    return ids

# Get the IDs of the training volumes that contain all necessary files


def get_train_test_ids_completed(mri_types, oversample_tumors=False, undersample_bg=False):
    train_ids = _get_ids_completed(train_dirs, mri_types, train=True)

    if undersample_bg or oversample_tumors:
        # Create sorted list of all volumes' fraction of tumour pixels
        frac_tumours_per_volume = sorted(
            [(id, _frac_tumours(id)) for id in train_ids], key=lambda x: x[1])

        # Undersample by removing 14 of the 100 lowest tumour fractions
        if undersample_bg:
            undersample_ids = [x[0] for x in frac_tumours_per_volume[:100]]
            for i in range(14):
                undersample_id = np.random.choice(undersample_ids)
                undersample_ids.remove(undersample_id)
                train_ids.remove(undersample_id)

        # Oversample by by randomly picking 98 times one of 25 highest tumour fraction volumes to duplicate
        if oversample_tumors:
            oversample_ids = [x[0] for x in frac_tumours_per_volume[-25:]]
            for i in range(98):
                oversample_id = np.random.choice(oversample_ids)
                train_ids.append(oversample_id)

    train_ids, test_ids = train_test_split(
        train_ids, test_size=0.2, random_state=42)
    return train_ids, test_ids

# Get the IDs of the validation volumes that contain all necessary files


def get_val_ids_completed(mri_types):
    return _get_ids_completed(validation_dirs, mri_types)


# Get the fraction of tumour pixels out of all pixels in a volume
def _frac_tumours(id):
    epsilon = 1e-6
    file_path_prefix = "BraTS20_Training"
    file_path = os.path.join(TRAIN_DATASET_PATH,
                             '{}_{}'.format(file_path_prefix, id),
                             '{}_{}'.format(file_path_prefix, id) + "_seg.nii")
    if os.path.exists(file_path):
        mask = nib.load(file_path).get_fdata()
        # Count the number of occurrences of each label
        labels, counts = np.unique(mask, return_counts=True)
        frac = 0
        if len(labels) > 1:
            bg_pixels = counts[0]
            if (labels[0] != 0):
                print('WARNING')
            tumour_pixels = counts[1:].sum()
            frac = (tumour_pixels + epsilon) / \
                ((tumour_pixels + bg_pixels) + epsilon)
        return frac


# Use data generator as data cannot be loaded all at once into RAM

# Implementation of Data Generator inspired by
# https://towardsdatascience.com/keras-data-generators-and-how-to-use-them-b69129ed779c
class DataGenerator(Sequence):
    'Generates data for Keras'

    def __init__(self, list_IDs, data_path, file_path_prefix, n_classes, modalities=['flair', 't1ce', 't2', 't1'], seed=-1, slice_range=100, slice_start=22,  slice_interval=1, to_fit=True, batch_size=1, dim=(128, 128), shuffle=True, augment=True):
        'Initialization'
        self.list_IDs = list_IDs
        self.data_path = data_path
        self.file_path_prefix = file_path_prefix
        self.n_channels = len(modalities)
        self.n_classes = n_classes
        self.modalities = modalities
        self.slice_range = slice_range
        self.slice_start = slice_start
        self.slice_interval = slice_interval if slice_interval >= 1 else 1
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.dim = dim
        self.shuffle = shuffle
        self.augment = augment
        self.slice_offset = 0
        if seed >= 0:
            np.random.seed = seed
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
        if self.slice_interval > 1:
            self.slice_offset = np.random.randint(0, self.slice_interval)

    def __getitem__(self, index):
        # Generate one batch of data

        # Generate indexes of the batch
        indexes = self.indexes[index *
                               self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[i] for i in indexes]

        return self._generate_data(list_IDs_temp)

    def _generate_data(self, list_IDs_temp):
        """Generates data containing batch_size images, and batch_size masks if to_fit
        :param list_IDs_temp: list of label ids to load
        :return: batch of images, and masks if to_fit
        """
        data = np.zeros((self.batch_size*self.slice_range //
                        self.slice_interval, self.dim[0], self.dim[1], self.n_channels))
        labels = np.zeros((self.batch_size*self.slice_range //
                          self.slice_interval, self.dim[0], self.dim[1], self.n_classes))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            scan_data = self._load_scan_data(ID)
            # include an offset so that distribution of slices is more randomized
            for j in range(self.slice_range//self.slice_interval - 1):
                modality_images = []
                for modality in self.modalities:
                    modality_images.append(
                        scan_data[modality][:, :, j*self.slice_interval+self.slice_offset+self.slice_start])

                if (self.to_fit):
                    masks = scan_data['seg'][:, :, j*self.slice_interval +
                                             self.slice_offset+self.slice_start]
                    # GD-ENHANCING is labelled as 4, but we change to 3 to make it easier to use iterate over each class
                    masks[masks == 4] = 3
                    if self.augment:
                        modality_images, masks = da.perform_data_augmentation(
                            modality_images, masks)
                    # Generate binary masks for each class
                    masks = cv2.resize(masks, (self.dim[0], self.dim[1]))
                    for c in range(self.n_classes):
                        labels[j + self.slice_range*i//self.slice_interval,
                               :, :, c] = (masks == c).astype(int)

                for chan, modality in enumerate(self.modalities):
                    data[j + self.slice_range*i//self.slice_interval, :, :,
                         chan] = cv2.resize(modality_images[chan], (self.dim[0], self.dim[1]))

        # Normalize data by dividing by brightest pixel
        data /= np.max(data)

        if self.to_fit:
            # When fitting, will use both images and masks
            return data, labels
        else:
            return data

    def _load_scan_data(self, scan_id):
        def file_path(i): return os.path.join(self.data_path,
                                              '{}_{}'.format(
                                                  self.file_path_prefix, i),
                                              '{}_{}'.format(self.file_path_prefix, i))
        # Load the .nii files for this scan
        # modality_files = [file_path(scan_id) + '_{}.nii'.format(modality for modality in MODALITIES)]

        scan_data = dict()
        for modality in self.modalities:
            modality_file = file_path(scan_id) + '_{}.nii'.format(modality)
            scan_data[modality] = nib.load(modality_file).get_fdata(
            ) if os.path.exists(modality_file) else None

        if self.to_fit:
            seg_file = file_path(scan_id) + '_seg.nii'
            scan_data['seg'] = nib.load(seg_file).get_fdata(
            ) if os.path.exists(seg_file) else None

        return scan_data


# Create generators for training and validation
def create_training_gen(train_ids, modalities, n_classes, batch_size, dim, slice_range, slice_start, slice_interval, augment=True, seed=-1):
    return DataGenerator(train_ids, TRAIN_DATASET_PATH, 'BraTS20_Training', n_classes, modalities, slice_range=slice_range, slice_start=slice_start, slice_interval=slice_interval, batch_size=batch_size, dim=dim, augment=augment, seed=seed)


def create_test_gen(test_ids, modalities, n_classes, batch_size, dim, slice_range, slice_start, slice_interval, augment=True, seed=-1):

    return DataGenerator(test_ids, TRAIN_DATASET_PATH, 'BraTS20_Training', n_classes, modalities, slice_range=slice_range, slice_start=slice_start, slice_interval=slice_interval, batch_size=batch_size, dim=dim, augment=augment, seed=seed)


def create_validation_gen(val_ids, modalities, n_classes, batch_size, dim, slice_range, slice_start, slice_interval, augment=True, seed=-1):
    return DataGenerator(val_ids, VALIDATION_DATASET_PATH, 'BraTS20_Validation', n_classes, modalities, slice_range=slice_range, slice_start=slice_start, slice_interval=slice_interval, batch_size=batch_size, dim=dim, augment=augment, seed=seed)
