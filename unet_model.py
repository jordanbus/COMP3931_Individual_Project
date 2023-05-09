from tensorflow.keras.callbacks import ReduceLROnPlateau
from dataset import get_train_test_ids_completed, create_training_gen, create_test_gen, create_validation_gen, get_val_ids_completed
from unet import create_unet
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from metrics import MulticlassMetrics as mcm, BinaryMetrics as bm
import os
import numpy as np
import cv2

BATCH_SIZE = 2
# Size to resize images to
IMG_SIZE = 128

SEGMENT_CLASSES = {
    0: 'NOT tumor',  # include background
    1: 'NECROTIC and NON-ENHANCING tumor core',
    2: 'Peritumoral EDEMA',
    3: 'GD-ENHANCING tumor',  # changed from original label 4
}
NUM_CLASSES = len(SEGMENT_CLASSES)

# Weights to use for each class in loss function
CLASS_WEIGHTS = [1, 10, 10, 10]

MODALITIES = ['flair', 't1ce']
IMG_CHANNELS = len(MODALITIES)
callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                      patience=2, min_lr=0.000001, verbose=1),
]


class UNetModel:
    def __init__(self,
                 models_path,
                 loss='combined_loss',
                 model_index=None,
                 modalities=MODALITIES,
                 segment_classes=SEGMENT_CLASSES,
                 class_weights=CLASS_WEIGHTS,
                 slice_interval=5,
                 slice_range=100,
                 slice_start=22,
                 seed=-1,
                 dummy_ids=False,
                 ):
        
        print("Constructing UnetModel.")

        self.seed = seed
        self.slice_interval = slice_interval
        self.slice_range = slice_range
        self.slice_start = slice_start
        self.n_channels = len(modalities)
        self.class_weights = class_weights
        self.segment_classes = segment_classes
        self.n_classes = len(segment_classes)
        self.loss = loss
        if self.n_classes == 1:
            print("Using binary metrics")
            self.metrics = ['accuracy',
                            bm.sensitivity,
                            bm.specificity,
                            bm.dice_loss,
                            bm.combined_loss_wrapper()]
            if loss == 'combined_loss':
                self.loss = bm.combined_loss_wrapper()
        else:
            print("Using multilabel metrics")
            self.metrics = ['accuracy',
                            mcm.sensitivity,
                            mcm.specificity,
                            mcm.binary_cross_entropy_per_channel_wrapper(self.class_weights),
                            mcm.dice_loss_wrapper(self.class_weights),
                            mcm.dice_coef_necrotic,
                            mcm.dice_coef_edema,
                            mcm.dice_coef_enhancing,
                            mcm.combined_loss_wrapper(self.class_weights)]
            if loss == 'combined_loss':
                self.loss = mcm.combined_loss_wrapper(self.class_weights)

        self.models_path = models_path
        # Categorical cross entropy loss requires softmax activation, whereas binary cross entropy requires sigmoid
        self.activation = 'softmax' if loss == 'categorical_crossentropy' else 'sigmoid'
        self.modalities = modalities
        # Check that modalities are valid
        for modality in modalities:
            if not modality in ['flair', 't1ce', 't1', 't2']:
                raise ValueError("Invalid modality: " + modality)
        # Check that loss function is valid 
        if isinstance(loss, str) and loss not in ['categorical_crossentropy', 'binary_crossentropy', 'combined_loss']:
            raise ValueError("Invalid loss function: " + loss)
        if not dummy_ids:
            self.train_ids, self.test_ids = get_train_test_ids_completed(
                mri_types=modalities)
            self.validation_ids = get_val_ids_completed(mri_types=modalities)
        else:
            self.train_ids = [i for i in range(1, 100)]
            self.test_ids = [i for i in range(100, 150)]
            self.validation_ids = [i for i in range(150, 200)]

        # If model index (index of model job) is specified, load the model for that job
        self.model = self.load_model(
            model_index) if model_index is not None else None

    # Compile the model using the loss function and metrics that this class was initialized with.
    def compile_model(self):
        if self.model is None:
            raise Exception("Cannot compile model before loading it")
        self.model.compile(loss=self.loss, optimizer=Adam(
            learning_rate=1e-3), metrics=self.metrics)

    # Load the weights from a saved model file.
    # Trainig is split into jobs which complete a number of epoch each, and model is saved with job index once these are complete.
    # Specifying job index will load the model from that job.
    # Can also specify a model path to load weights from.
    # Set compile to False if you want to recompile the model later with metrics and loss function that this class was intialized with.
    def load_model(self, job_index=None, model_path=None, compile=True):
        if model_path is None and job_index is None:
            raise Exception("Must provide either a model path or a job index")

        # Get the path of the correct job if job index is specified and not model path
        model_path = model_path if model_path is not None else os.path.join(
            self.models_path, "model_job{}".format(job_index))

        # Need to add custom metric and loss functions to custom objects when loading model
        if self.n_classes > 1:
            custom_objects = {
                "combined_loss": mcm.combined_loss_wrapper(self.class_weights),
                "dice_loss": mcm.dice_loss_wrapper(self.class_weights),
                "dice_coef_necrotic": mcm.dice_coef_necrotic,
                "dice_coef_edema": mcm.dice_coef_edema,
                "dice_coef_enhancing": mcm.dice_coef_enhancing,
                "sensitivity": mcm.sensitivity,
                "specificity": mcm.specificity,
                "binary_cross_entropy_per_channel": mcm.binary_cross_entropy_per_channel_wrapper(self.class_weights),
            }
        else:
            custom_objects = {
                "combined_loss": bm.combined_loss_wrapper(),
                "dice_loss": bm.dice_loss,
                "sensitivity": bm.sensitivity,
                "specificity": bm.specificity,
            }

        self.model = load_model(
            model_path, custom_objects=custom_objects, compile=compile)

    # Starts or resumes training of the model for job's portion of total n_epochs
    def train_model(self, job_index, n_jobs, n_epochs, batch_size=BATCH_SIZE):
        for i in range(job_index, n_jobs):
            job_index = i

            # Calculate the start and end epoch for this job
            # to display the correct number of epochs that the model has been trained for
            start_epoch = job_index * n_epochs // n_jobs
            end_epoch = (job_index + 1) * n_epochs // n_jobs

            # Load the model from a file
            model_filename = os.path.join(
                self.models_path, "model_job{}".format(job_index))
            if start_epoch == 0:
                # Create a new model if this is the first job
                inputs = Input((IMG_SIZE, IMG_SIZE, self.n_channels))
                self.model = create_unet(
                    inputs, activation=self.activation, num_classes=self.n_classes, loss=self.loss, metrics=self.metrics)
            else:
                if self.model is None:
                    # Load previous model to continue training
                    load_model(job_index-1)
                    
        if self.model is None:
            raise Exception("Unable to load a model to train")
        
        # If softmax activation is used, one-hot encode the labels in the data generators
        one_hot = self.activation == 'softmax'
        training_gen = create_training_gen(self.train_ids, one_hot=one_hot, slice_range=self.slice_range, slice_start=self.slice_start, slice_interval=self.slice_interval,
                                           modalities=self.modalities, batch_size=batch_size, dim=(IMG_SIZE, IMG_SIZE), segment_classes=self.segment_classes,seed=self.seed)
        test_gen = create_test_gen(self.test_ids, one_hot=one_hot, slice_range=self.slice_range, slice_start=self.slice_start, slice_interval=self.slice_interval,
                                   modalities=self.modalities, batch_size=batch_size, dim=(IMG_SIZE, IMG_SIZE), segment_classes=self.segment_classes, seed=self.seed)

        # Start training
        self.model.fit(training_gen,
                       epochs=end_epoch,
                       steps_per_epoch=len(self.train_ids)/batch_size,
                       initial_epoch=start_epoch,
                       callbacks=callbacks,
                       validation_data=test_gen)

        # Save the model to a file
        self.model.save(model_filename)

    # Evaluates the model on the test set, giving metric scores
    def evaluate_model(self, batch_size=BATCH_SIZE):
        if self.model is None:
            raise Exception("No model to evaluate")
        test_gen = create_test_gen(self.test_ids, slice_range=self.slice_range, slice_start=self.slice_start, slice_interval=self.slice_interval,
                                   modalities=self.modalities, batch_size=batch_size, dim=(IMG_SIZE, IMG_SIZE), segment_classes=self.segment_classes, seed=self.seed)
        self.model.evaluate(test_gen, steps=len(self.test_ids)/batch_size)

    # Predicts the segmentation masks of the validation set
    def validation_predictions(self, batch_size=BATCH_SIZE, slice_interval=None):
        if slice_interval is None:
            slice_interval = self.slice_interval
        if self.model is None:
            raise Exception("No model to predict with")
        val_gen = create_validation_gen(self.validation_ids, slice_range=self.slice_range, slice_start=self.slice_start, slice_interval=slice_interval,
                                        modalities=self.modalities, batch_size=batch_size, dim=(IMG_SIZE, IMG_SIZE), segment_classes=self.segment_classes, seed=self.seed)
        self.model.predict(val_gen, steps=len(self.validation_ids)/batch_size)

    # Predicts the segmentation masks of the given images
    def predict(self, images):
        if self.model is None:
            raise Exception("No model to predict with")
        elif images.shape[-1] != self.n_channels:
            raise Exception("Number of channels in images does not match the model")
        elif not len(images) > 0:
            raise Exception("No images to predict")
        
        X = np.empty(images.shape)
        for i in range(images.shape[0]):
            for chan in range(self.n_channels):
                X[i, ..., chan] = cv2.resize(
                    images[i, ..., chan], (IMG_SIZE, IMG_SIZE))

        # Normalize the images same way as in data generator, by brightest pixel in the data
        X = X/np.max(X)
        return self.model.predict(X)
