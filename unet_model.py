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
IMG_SIZE = 128
SEGMENT_CLASSES = {
    0: 'NOT tumor',  # include background
    1: 'NECROTIC and NON-ENHANCING tumor core',
    2: 'Peritumoral EDEMA',
    3: 'GD-ENHANCING tumor',  # labelled as 4 in dataset - we change to 3
}
NUM_CLASSES = len(SEGMENT_CLASSES)

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
                 seed=-1
                 ):
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
        self.activation = 'softmax' if loss == 'categorical_crossentropy' else 'sigmoid'
        self.modalities = modalities

        self.train_ids, self.test_ids = get_train_test_ids_completed(
            mri_types=modalities)
        self.validation_ids = get_val_ids_completed(mri_types=modalities)

        self.model = self.load_model(
            model_index) if model_index is not None else None

    def compile_model(self):
        self.model.compile(loss=self.loss, optimizer=Adam(
            learning_rate=1e-3), metrics=self.metrics)

    def load_model(self, job_index=None, model_path=None, compile=True):
        if model_path is None and job_index is None:
            raise Exception("Must provide either a model path or a job index")

        model_path = model_path if model_path is not None else os.path.join(
            self.models_path, "model_job{}".format(job_index))

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

    # Starts or resumes training of the model for job's fraction of total n_epochs
    def train_model(self, job_index, n_jobs, n_epochs, batch_size=BATCH_SIZE):
        for i in range(job_index, n_jobs):
            job_index = i

            start_epoch = job_index * n_epochs // n_jobs
            end_epoch = (job_index + 1) * n_epochs // n_jobs

            # Load the model from a file
            model_filename = os.path.join(
                self.models_path, "model_job{}".format(job_index))
            if start_epoch == 0:
                inputs = Input((IMG_SIZE, IMG_SIZE, self.n_channels))
                self.model = create_unet(
                    inputs, activation=self.activation, num_classes=self.n_classes, loss=self.loss, metrics=self.metrics)
            else:
                if self.model is None:
                    # Load previous model to continue training
                    load_model(job_index-1)

        one_hot = self.activation == 'softmax'
        training_gen = create_training_gen(self.train_ids, one_hot=one_hot, slice_range=self.slice_range, slice_start=self.slice_start, slice_interval=self.slice_interval,
                                           modalities=self.modalities, batch_size=batch_size, dim=(IMG_SIZE, IMG_SIZE), segment_classes=self.segment_classes,seed=self.seed)
        test_gen = create_test_gen(self.test_ids, one_hot=one_hot, slice_range=self.slice_range, slice_start=self.slice_start, slice_interval=self.slice_interval,
                                   modalities=self.modalities, batch_size=batch_size, dim=(IMG_SIZE, IMG_SIZE), segment_classes=self.segment_classes, seed=self.seed)

        self.model.fit(training_gen,
                       epochs=end_epoch,
                       steps_per_epoch=len(self.train_ids)/batch_size,
                       initial_epoch=start_epoch,
                       callbacks=callbacks,
                       validation_data=test_gen)

        # Save the model to a file
        self.model.save(model_filename)

    # Evaluates the model on the test set
    def evaluate_model(self, batch_size=BATCH_SIZE):
        if self.model is None:
            print("Model not loaded")
        test_gen = create_test_gen(self.test_ids, slice_range=self.slice_range, slice_start=self.slice_start, slice_interval=self.slice_interval,
                                   modalities=self.modalities, batch_size=batch_size, dim=(IMG_SIZE, IMG_SIZE), segment_classes=self.segment_classes, seed=self.seed)
        self.model.evaluate(test_gen, steps=len(self.test_ids)/batch_size)

    def validation_predictions(self, batch_size=BATCH_SIZE, slice_interval=None):
        if slice_interval is None:
            slice_interval = self.slice_interval
        if self.model is None:
            print("Model not loaded")
        val_gen = create_validation_gen(self.validation_ids, slice_range=self.slice_range, slice_start=self.slice_start, slice_interval=slice_interval,
                                        modalities=self.modalities, batch_size=batch_size, dim=(IMG_SIZE, IMG_SIZE), segment_classes=self.segment_classes, seed=self.seed)
        self.model.predict(val_gen, steps=len(self.validation_ids)/batch_size)

    # Predicts the segmentation of the given images
    def predict(self, images):
        X = np.empty(images.shape)
        for i in range(images.shape[0]):
            for chan in range(self.n_channels):
                X[i, ..., chan] = cv2.resize(
                    images[i, ..., chan], (IMG_SIZE, IMG_SIZE))

        X = X/np.max(X)
        return self.model.predict(X)
