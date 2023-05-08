import os
import re
import nibabel as nib
import numpy as np
import cv2
import data_augmentation as da
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split
import tensorflow as tf

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

    # Use same split every time
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

    def __init__(self, list_IDs, data_path, file_path_prefix, segment_classes, modalities=['flair', 't1ce', 't2', 't1'], seed=-1, slice_range=100, slice_start=22,  slice_interval=1, to_fit=True, batch_size=1, dim=(128, 128), shuffle=True, augment=True, one_hot=False):
        'Initialization'
        self.list_IDs = list_IDs
        self.data_path = data_path
        self.file_path_prefix = file_path_prefix
        self.n_channels = len(modalities)
        self.segment_classes = segment_classes
        self.n_classes = len(segment_classes)
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
        # if n_classes is one, assume binary classifier
        self.classifier = 'binary' if self.n_classes == 1 else 'multilabel'
        self.one_hot = one_hot
        if seed >= 0:
            np.random.seed = seed
        # Shuffle the data and calculate slice offset for initial epoch
        self.on_epoch_end()

    # Gets the number of batches per epoch
    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    # Shuffle the data (by shuffling the scan IDs) after each epoch
    #.. and calculate a new slice offset to use for the next epoch
    def on_epoch_end(self):
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

    def _create_single_binary_mask(self, mask, labels, slice_index, scan_id):
        # Generate a single binary mask for whole tumour
        labels[slice_index + self.slice_range*scan_id //
               self.slice_interval, :, :] = (mask > 0).astype(int)
        return labels

    def _create_one_hot_encoded_mask(self, masks, labels, slice_index, scan_id):
        # Generate one-hot encoded masks
        # Change GD-Enhancing to label 3 from 4
        masks[masks == 4] = 3
        labels[slice_index + self.slice_range*scan_id // self.slice_interval,
               :, :] = tf.one_hot(masks, self.n_classes)
        return labels

    def _create_binary_mask_per_class(self, masks, labels, slice_index, scan_id):
        # Generate binary masks for each class
        for c in self.segment_classes.keys():
            labels[slice_index + self.slice_range*scan_id//self.slice_interval,
                   :, :, c] = (masks == c).astype(int)
        
        return labels

    # Generate a batch of pre-processed data, along with ground truth masks if to_fit,
    #.. using the list of IDs for the volumes to include the slices from.
    # Batches will contain scan images, with each modality in separate channels, 
    #.. and ground truth masks will be binary masks indicating the different segment classes.
    def _generate_data(self, list_IDs_temp):

        data = np.zeros((self.batch_size*self.slice_range //
                        self.slice_interval, self.dim[0], self.dim[1], self.n_channels))
        labels = np.zeros((self.batch_size*self.slice_range //
                          self.slice_interval, self.dim[0], self.dim[1], self.n_classes))

        # Binary classifier only needs one channel for ground truth mask
        if self.classifier == 'binary':
            labels = np.zeros((self.batch_size*self.slice_range //
                               self.slice_interval, self.dim[0], self.dim[1]))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            scan_data = self._load_scan_data(ID)
            # Include an offset so that distribution of slices is more randomized
            # Hence need for -1 to not go out of bounds (slice offset should be less than slice interval)
            for j in range(self.slice_range//self.slice_interval - 1):
                modality_images = []
                for modality in self.modalities:
                    modality_images.append(
                        scan_data[modality][:, :, j*self.slice_interval+self.slice_offset+self.slice_start])

                if (self.to_fit):
                    masks = scan_data['seg'][:, :, j*self.slice_interval +
                                             self.slice_offset+self.slice_start]
                    if self.augment:
                        modality_images, masks = da.perform_data_augmentation(
                            modality_images, masks)
                    # Generate masks according to classifier type, resize to specified dimensions
                    masks = cv2.resize(masks, (self.dim[0], self.dim[1]))
                    if self.classifier == 'binary':
                        labels = self._create_single_binary_mask(
                            masks, labels, j, i)
                    elif self.classifier == 'multilabel':
                        if self.one_hot:
                            labels = self._create_one_hot_encoded_mask(
                                masks, labels, j, i)
                        else:
                            labels = self._create_binary_mask_per_class(
                                masks, labels, j, i)

                # Add the slices for each modality to the data array, resized to the specified dimensions
                for chan, modality in enumerate(self.modalities):
                    data[j + self.slice_range*i//self.slice_interval, :, :,
                         chan] = cv2.resize(modality_images[chan], (self.dim[0], self.dim[1]))

        # Normalize batch of data by dividing by brightest pixel in batch
        data /= np.max(data)

        if self.to_fit:
            # When fitting, will use both images and masks
            return data, labels
        else:
            return data

    # Get the image (and mask if to_fit) data for specified scan, given the scan ID
    def _load_scan_data(self, scan_id):
        def file_path(i): return os.path.join(self.data_path,
                                              '{}_{}'.format(
                                                  self.file_path_prefix, i),
                                              '{}_{}'.format(self.file_path_prefix, i))

        # Store each modality in a dictionary, where the modality type is the key
        scan_data = dict()
        for modality in self.modalities:
            modality_file = file_path(scan_id) + '_{}.nii'.format(modality)
            scan_data[modality] = nib.load(modality_file).get_fdata(
            ) if os.path.exists(modality_file) else None

        #  Include the ground truth mask in the dictionary if to_fit
        if self.to_fit:
            seg_file = file_path(scan_id) + '_seg.nii'
            scan_data['seg'] = nib.load(seg_file).get_fdata(
            ) if os.path.exists(seg_file) else None

        return scan_data


# Create generators for training and validation, including the correct paths for each set
def create_training_gen(train_ids, modalities, segment_classes, batch_size, dim, slice_range, slice_start, slice_interval, one_hot=False, augment=True, seed=-1, shuffle=True):
    return DataGenerator(train_ids, TRAIN_DATASET_PATH, 'BraTS20_Training', segment_classes, modalities, one_hot=one_hot, slice_range=slice_range, slice_start=slice_start, slice_interval=slice_interval, batch_size=batch_size, dim=dim, augment=augment, seed=seed, shuffle=shuffle)


def create_test_gen(test_ids, modalities, segment_classes, batch_size, dim, slice_range, slice_start, slice_interval, one_hot=False, seed=-1, shuffle=True):
    return DataGenerator(test_ids, TRAIN_DATASET_PATH, 'BraTS20_Training', segment_classes, modalities, one_hot=one_hot, slice_range=slice_range, slice_start=slice_start, slice_interval=slice_interval, batch_size=batch_size, dim=dim, augment=False, seed=seed, shuffle=shuffle)


def create_validation_gen(val_ids, modalities, segment_classes, batch_size, dim, slice_range, slice_start, slice_interval, seed=-1, shuffle=True):
    return DataGenerator(val_ids, VALIDATION_DATASET_PATH, 'BraTS20_Validation', segment_classes, modalities, slice_range=slice_range, slice_start=slice_start, slice_interval=slice_interval, batch_size=batch_size, dim=dim, to_fit=False, seed=seed, shuffle=shuffle)
