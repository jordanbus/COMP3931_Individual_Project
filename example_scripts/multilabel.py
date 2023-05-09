import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, BatchNormalization, ReLU, Concatenate, Conv2DTranspose
import tensorflow.keras.backend as K
import re
from tensorflow.keras.models import load_model
import nibabel as nib
from tensorflow.keras.utils import Sequence
import cv2
import numpy as np
from tensorflow.keras.callbacks import ReduceLROnPlateau, CSVLogger
import sys
from sklearn.model_selection import train_test_split


job_index = int(sys.argv[1])
num_jobs = int(sys.argv[2])
num_epochs = int(sys.argv[3])

USERDIR = '/nobackup/sc20jwb'

# Define the directories where the scans are located
TRAIN_DATASET_PATH = USERDIR +'/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/'
VALIDATION_DATASET_PATH = USERDIR + '/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData'


VOLUME_SLICES = 100
VOLUME_START_AT = 22
IMG_SIZE = 128
BATCH_SIZE = int(sys.argv[4])
SLICE_INTERVAL = 5
SEED = int(sys.argv[5])
np.random.seed(SEED)
DROPOUT = float(sys.argv[6])
SEGMENT_CLASSES = {
    0 : 'NOT tumor', # include background
    1 : 'NECROTIC and NON-ENHANCING tumor core',
    2 : 'Peritumoral EDEMA',
    3 : 'GD-ENHANCING tumor', # labelled as 4 in dataset - we change to 3
}
NUM_CLASSES = len(SEGMENT_CLASSES)
UNDERSAMPLE_MAJORITY = int(sys.argv[7]) == 1
OVERSAMPLE_MINORITY = int(sys.argv[8]) == 1
MODALITIES = []
for i in range(9,len(sys.argv)):
  MODALITIES.append(sys.argv[i])
IMG_CHANNELS = len(MODALITIES)
print(MODALITIES)

def create_unet(inputs, loss="categorical_crossentropy", num_filters=32, num_classes=NUM_CLASSES, kernel_initializer='he_normal', dropout_rate=0.2, metrics=['accuracy']):
    def encode_conv(inputs, filters):
        conv = Conv2D(filters, (3,3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(inputs)
#        conv = BatchNormalization()(conv)  # Add batch normalization layer
        conv = Dropout(dropout_rate)(conv)
        conv = Conv2D(filters, (3,3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(conv)
#        conv = BatchNormalization()(conv)  # Add batch normalization layer
        pool = MaxPooling2D(pool_size=(2, 2))(conv)
        return conv,pool

    def bottleneck_conv(inputs, filters):
        conv = Conv2D(filters, (3,3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(inputs)
#        conv = BatchNormalization()(conv)  # Add batch normalization layer
        conv = Dropout(dropout_rate)(conv)
        conv = Conv2D(filters, (3,3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(conv)
#        conv = BatchNormalization()(conv)  # Add batch normalization layer
        return conv

    def decode_conv(inputs, skip_connection, filters, concat_axis=-1):
        up = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(inputs)
        up = Concatenate(axis=concat_axis)([up, skip_connection])
        conv = Conv2D(filters, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(up)
#        conv = BatchNormalization()(conv)  # Add batch normalization layer
        conv = Dropout(dropout_rate)(conv)
        conv = Conv2D(filters, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(conv)
#        conv = BatchNormalization()(conv)  # Add batch normalization layer
        return conv

    conv1, pool1 = encode_conv(inputs, num_filters)
    conv2, pool2 = encode_conv(pool1, num_filters*2)
    conv3, pool3 = encode_conv(pool2, num_filters*4)
    conv4, pool4 = encode_conv(pool3, num_filters*8)
    conv5 = bottleneck_conv(pool4, num_filters*16)
    up6 = decode_conv(conv5, conv4, num_filters*8)
    up7 = decode_conv(up6, conv3, num_filters*4)
    up8 = decode_conv(up7, conv2, num_filters*2)
    up9 = decode_conv(up8, conv1, num_filters, concat_axis=3)

    outputs = Conv2D(num_classes, (1,1), activation='sigmoid')(up9)

    # Define the model
    model = Model(inputs=inputs, outputs=outputs)
    
    model.compile(loss=loss, optimizer=Adam(learning_rate=1e-3), metrics = metrics)

    return model

#@title Custom metrics

def binary_cross_entropy_per_channel(y_true, y_pred):
    # Define class weights
    class_weights = np.array([1.0, 100.0, 100.0, 100.0])
    class_weights /= np.sum(class_weights)

    # Compute binary cross-entropy loss for each channel separately, with class weights
    loss = 0.0
    channels = y_pred.shape[-1]
    for i in range(channels):
        weighted_loss = class_weights[i] * tf.losses.binary_crossentropy(y_true[..., i], y_pred[..., i])
        loss += weighted_loss
    return loss

# - Specificity
def specificity(y_true, y_pred):
    # specificity = TN / ( TN + FP )
    # Get sum of intersection of predicted negatives with ground truth negatives for true negatives (TN)
    tn = K.sum((1-y_true[...,1:]) * (1-K.round(y_pred[...,1:])))
    # Get sum of ground truth negatives for actual negatives (equal to TN + FP)
    tn_fp = K.sum(1-y_true[...,1:])
    return tn / (tn_fp + K.epsilon())

# - Sensitivity
def sensitivity(y_true, y_pred):
    # sensitivity = TP / ( TP + FN )
    # Get sum of intersection of predicted positives with ground truth positives for true positives (TP)
    tp = K.sum(y_true[...,1:] * K.round(y_pred[...,1:]))
    # Get sum of ground truth positives for actual positives (equal to TP + FN)
    tp_fn = K.sum(y_true[...,1:])
    return tp / (tp_fn + K.epsilon())

# - Dice coefficient index (*Dci*)
def _dice_loss_for_label(y_true, y_pred, label, smooth=1e-7):
    # Flatten the ground truth and predicted masks
    y_true_f = K.flatten(y_true[...,label])
    y_pred_f = K.flatten(y_pred[...,label])

    # Compute the intersection and union
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f)

    # Compute the Dice coefficient
    dice_coef = (2.0 * intersection + smooth) / (union + smooth)

    # Compute the Dice loss as 1 - Dice coefficient
    dice_loss = 1.0 - dice_coef

    return dice_loss

def dice_loss(y_true, y_pred, smooth=1e-6):
    # Define class weights
    class_weights = np.array([1.0, 100.0, 100.0, 100.0])  # Adjust the weights as needed
    class_weights /= np.sum(class_weights)

    dice = 0
    for i in range(NUM_CLASSES):
        dice += class_weights[i] *  _dice_loss_for_label(y_true, y_pred, i, smooth)
    return dice

 
def dice_coef_necrotic(y_true, y_pred, smooth=1e-6):
    return 1 - _dice_loss_for_label(y_true, y_pred, 1, smooth)

def dice_coef_edema(y_true, y_pred, smooth=1e-6):
    return 1 - _dice_loss_for_label(y_true, y_pred, 2, smooth)


def dice_coef_enhancing(y_true, y_pred, smooth=1e-6):
    return 1 - _dice_loss_for_label(y_true, y_pred, 3, smooth)

def combined_loss(y_true, y_pred, alpha=0.5):
    """
    Combined loss function that combines binary cross entropy and Dice loss.
    :param y_true: Ground truth labels.
    :param y_pred: Predicted labels.
    :param alpha: Weight of binary cross entropy in the loss.
    :return: combined loss.
    """
    # Calculate binary cross entropy
    bce = binary_cross_entropy_per_channel(y_true, y_pred)

    # Calculate Dice loss
    dl = dice_loss(y_true, y_pred)

    # Calculate combined loss
    combined = alpha * bce + (1-alpha) * dl

    return combined

train_dirs = [f.path for f in os.scandir(TRAIN_DATASET_PATH) if f.is_dir()]
validation_dirs = [f.path for f in os.scandir(VALIDATION_DATASET_PATH) if f.is_dir()]

def get_id_from_path(path):
  m = re.search('BraTS20_(Training|Validation)_(\d+)', path)
  return m.group(2)

def get_ids_to_remove(directories, look_for):
  delete_ids = []
  for directory in directories:
    for _,__,files in os.walk(directory):
      # print(files)
      # print()
      delete = False
      for word in look_for:
        if not any(word in x for x in files):
            delete = True
            break
      if delete:

        id = get_id_from_path(directory)
        delete_ids.append(id)
  return delete_ids

# Get IDs for volumes and remove volume if doesn't contain all necessary files
delete_ids = get_ids_to_remove(train_dirs, ['t1ce', 'flair', 'seg'])
train_ids = list(set([get_id_from_path(p) for p in train_dirs]))
print('Ignoring training volumes: ' + ",".join(delete_ids))
for id in delete_ids:
  train_ids.remove(id)

delete_ids = get_ids_to_remove(validation_dirs, ['t1ce', 'flair'])
validation_ids = list(set([get_id_from_path(p) for p in validation_dirs]))
print('Ignoring validation volumes: ' + ",".join(delete_ids))
for id in delete_ids:
  validation_ids.remove(id)

train_ids, test_ids = train_test_split(train_ids,test_size=0.2, random_state=42)

def frac_tumours(id):
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
      frac = (tumour_pixels + epsilon) / ((tumour_pixels + bg_pixels) + epsilon)
    return frac
  
frac_tumours_per_volume = sorted([(id, frac_tumours(id)) for id in train_ids], key=lambda x: x[1])
if UNDERSAMPLE_MAJORITY:
    undersample_ids = [x[0] for x in frac_tumours_per_volume[:100]]
    for i in range(25):
        undersample_id = np.random.choice(undersample_ids)
        undersample_ids.remove(undersample_id)
        train_ids.remove(undersample_id)

if OVERSAMPLE_MINORITY:
    oversample_ids = [x[0] for x in frac_tumours_per_volume[-25:]]
    for i in range(100):
        oversample_id = np.random.choice(oversample_ids)
        train_ids.append(oversample_id)
# print(len(train_ids))

def perform_data_augmentation(images, mask):
    """
    Apply data augmentation techniques to the input images and mask.
    """

    # Randomly select an augmentation technique
    augmentation_type = np.random.choice(['rotate', 'flip_horizontal', 'flip_vertical', 'adjust_brightness', 'zoom'])
    # Check if mask contains tumor classes
    if np.max(mask) > 0:
        augmented_images = []
        # Apply augmentation only to slices containing tumor classes
        if augmentation_type == 'rotate':
            # Apply rotation
            angle = np.random.uniform(-15, 15)
            for image in images:
              augmented_images.append(rotate_image(image, angle))
            mask = rotate_image(mask, angle)

        elif augmentation_type == 'flip_horizontal':
            # Apply horizontal flip
            for image in images:
              augmented_images.append(cv2.flip(image, 1))
            mask = cv2.flip(mask, 1)
        elif augmentation_type == 'flip_vertical':
            # Apply vertical flip
            for image in images:
              augmented_images.append(cv2.flip(image, 0))
            mask = cv2.flip(mask, 0)
        elif augmentation_type == 'adjust_brightness':
            # Apply brightness adjustment
            brightness_factor = np.random.uniform(0.8, 1.2)
            for image in images:
              augmented_images.append(adjust_brightness(image, brightness_factor))
        elif augmentation_type == 'zoom':
            # Random zoom
            zoom_factor = np.random.uniform(0.9, 1.1)
            for image in images:
              augmented_images.append(zoom_image(image, zoom_factor))
            mask = zoom_image(mask, zoom_factor)
        
        return augmented_images, mask

    # Return original images and mask
    return images, mask


def rotate_image(image, angle):
    """
    Rotate the image by given angle.
    """
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    return cv2.warpAffine(image, M, (cols, rows))

def zoom_image(image, factor):
    """
    Zoom the image by given factor.
    """
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 0, factor)
    return cv2.warpAffine(image, M, (cols, rows))

def adjust_brightness(image, factor):
    """
    Adjust the brightness of the image by given factor.
    """
    # Convert image to float32 for arithmetic operations
    image = image.astype(np.float32)

    # Adjust brightness of each channel separately
    for i in range(image.shape[-1]):
        image[..., i] = image[..., i] * factor

    # Clip pixel values to [0, 255]
    image = np.clip(image, 0, 255)

    # Convert image back to uint8
    image = image.astype(np.uint8)

    return image

#@title Data Generator
# https://towardsdatascience.com/keras-data-generators-and-how-to-use-them-b69129ed779c

# Use data generator as data cannot be loaded all at once into RAM
class DataGenerator(Sequence):
  'Generates data for Keras'
  def __init__(self, list_IDs, data_path, file_path_prefix, seed=-1, slice_interval=1, to_fit=True, batch_size=BATCH_SIZE, dim=(IMG_SIZE,IMG_SIZE), shuffle=True, augment=True):
      'Initialization'
      self.list_IDs = list_IDs
      self.data_path = data_path
      self.file_path_prefix = file_path_prefix
      self.slice_interval = slice_interval if slice_interval >=1 else 1
      self.to_fit = to_fit
      self.batch_size = batch_size
      self.dim = dim
      self.n_channels = IMG_CHANNELS
      self.shuffle = shuffle
      self.augment = augment
      self.slice_offset=0
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
      indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

      # Find list of IDs
      list_IDs_temp = [self.list_IDs[i] for i in indexes]

      return self._generate_data(list_IDs_temp)

  def _generate_data(self, list_IDs_temp):
      """Generates data containing batch_size images, and batch_size masks if to_fit
      :param list_IDs_temp: list of label ids to load
      :return: batch of images, and masks if to_fit
      """
      data = np.zeros((self.batch_size*VOLUME_SLICES//self.slice_interval, IMG_SIZE, IMG_SIZE, self.n_channels))
      labels = np.zeros((self.batch_size*VOLUME_SLICES//self.slice_interval, IMG_SIZE, IMG_SIZE, NUM_CLASSES))

      # Generate data
      for i, ID in enumerate(list_IDs_temp):
          scan_data = self._load_scan_data(ID)
          # include an offset so that distribution of slices is more randomized
          for j in range(VOLUME_SLICES//self.slice_interval - 1):
              modality_images = []
              for modality in MODALITIES:
                modality_images.append(scan_data[modality][:,:,j*self.slice_interval+self.slice_offset+VOLUME_START_AT])
              
              if (self.to_fit):
                masks = scan_data['seg'][:,:,j*self.slice_interval+self.slice_offset+VOLUME_START_AT]
                masks[masks==4] = 3
                if self.augment:
                    modality_images, masks = perform_data_augmentation(modality_images, masks)
                # segmentation classes
                masks = cv2.resize(masks, (IMG_SIZE, IMG_SIZE))
                for c in range(NUM_CLASSES):
                  labels[j + VOLUME_SLICES*i//self.slice_interval,:,:,c] = (masks == c).astype(int)
              
              for chan, modality in enumerate(MODALITIES):
                data[j + VOLUME_SLICES*i//self.slice_interval,:,:,chan] = cv2.resize(modality_images[chan], (IMG_SIZE, IMG_SIZE))

      data /= np.max(data)

      if self.to_fit:
        # Generate masks
        return data, labels
      else:
        return data

  def _load_scan_data(self, scan_id):
      file_path = lambda i: os.path.join(self.data_path, 
                                         '{}_{}'.format(self.file_path_prefix, i), 
                                         '{}_{}'.format(self.file_path_prefix, i))
      # Load the .nii files for this scan
      scan_data = dict()
      for modality in MODALITIES:
          modality_file = file_path(scan_id) + '_{}.nii'.format(modality)
          scan_data[modality] = nib.load(modality_file).get_fdata() if os.path.exists(modality_file) else None

      if self.to_fit:
        seg_file = file_path(scan_id) + '_seg.nii'
        scan_data['seg'] = nib.load(seg_file).get_fdata() if os.path.exists(seg_file) else None
      
      return scan_data


start_epoch = job_index * num_epochs // num_jobs
end_epoch = (job_index + 1) * num_epochs // num_jobs

# Load the model from a file

content_path = os.path.join(USERDIR, "final_multilabel", "_".join(MODALITIES))
if not os.path.exists(content_path):
  os.mkdir(content_path)
  
models_path = os.path.join(content_path, 'models')
if not os.path.exists(models_path):
  os.mkdir(models_path)
logs_path = os.path.join(content_path, 'training_logs')
if not os.path.exists(logs_path):
  os.mkdir(logs_path)
  
training_log_path = os.path.join(logs_path, 'training_log_job' + str(job_index) + '.csv')
callbacks = [
  # EarlyStopping(monitor='loss', min_delta=0, patience=2, verbose=1, mode='auto'),
  CSVLogger(training_log_path, separator=",", append=False),
  ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5, verbose=1),
]

prev_model_filename = os.path.join(models_path,"model_job" + str(job_index - 1))
model_filename = os.path.join(models_path, "model_job" + str(job_index))
if start_epoch == 0:
    metrics = ['accuracy', sensitivity, specificity, binary_cross_entropy_per_channel, dice_loss, dice_coef_necrotic, dice_coef_edema, dice_coef_enhancing]
    inputs = Input((IMG_SIZE,IMG_SIZE,IMG_CHANNELS))
    model = create_unet(inputs, loss=combined_loss, metrics=metrics, dropout_rate=DROPOUT)
else:
    model = load_model(prev_model_filename, custom_objects={
        "combined_loss": combined_loss,
        "dice_loss": dice_loss,
        "dice_coef_necrotic": dice_coef_necrotic,
        "dice_coef_edema": dice_coef_edema,
        "dice_coef_enhancing": dice_coef_enhancing, 
        "sensitivity": sensitivity,
        "specificity": specificity,
	"binary_cross_entropy_per_channel": binary_cross_entropy_per_channel
        })

training_gen = DataGenerator(train_ids, TRAIN_DATASET_PATH, 'BraTS20_Training', slice_interval=SLICE_INTERVAL)
test_gen = DataGenerator(test_ids, TRAIN_DATASET_PATH, 'BraTS20_Training', augment=False)

model.fit(training_gen, 
          epochs=end_epoch, 
          steps_per_epoch=len(train_ids)/BATCH_SIZE,
          initial_epoch=start_epoch, 
          callbacks=callbacks, 
          validation_data = test_gen)

# Save the model to a file
model.save(model_filename)

