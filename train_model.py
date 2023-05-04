from tensorflow.keras.callbacks import ReduceLROnPlateau
from dataset import get_train_test_ids_completed, create_training_gen, create_test_gen
from unet import create_unet
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from metrics import MultilabelMetrics, BinaryMetrics
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
train_ids, test_ids = get_train_test_ids_completed(mri_types=['flair', 't1ce'])

callbacks = [
    # EarlyStopping(monitor='loss', min_delta=0, patience=2, verbose=1, mode='auto'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                      patience=2, min_lr=0.000001, verbose=1),
]


class UNetModel:
    def __init__(self, 
                models_path, 
                model_index=None, 
                activation='sigmoid',
                modalities=MODALITIES, 
                n_classes=NUM_CLASSES,
                class_weights=CLASS_WEIGHTS,
                 ):
        self.n_channels = len(modalities)
        self.class_weights = class_weights
        self.n_classes = n_classes
        if self.n_classes == 1:
            print("Using binary metrics")
            self.metric_funcs = BinaryMetrics()
            self.metrics = ['accuracy',
                            self.metric_funcs.sensitivity,
                            self.metric_funcs.specificity,]
        else:
            print("Using multilabel metrics")
            self.metric_funcs = MultilabelMetrics(class_weights)
            self.metrics = ['accuracy',
                        self.metric_funcs.sensitivity,
                        self.metric_funcs.specificity,
                        self.metric_funcs.binary_cross_entropy_per_channel,
                        self.metric_funcs.dice_loss,
                        self.metric_funcs.dice_coef_necrotic,
                        self.metric_funcs.dice_coef_edema,
                        self.metric_funcs.dice_coef_enhancing]
        self.models_path = models_path
        self.activation = activation
        self.modalities = modalities
        
        self.model = self.load_model(
            model_index) if model_index is not None else None

    def compile_model(self):
        loss = 'categorical_crossentropy' if self.activation == 'softmax' else self.metric_funcs.combined_loss
        self.model.compile(loss=loss, optimizer=Adam(
            learning_rate=1e-3), metrics=self.metrics)

    def load_model(self, job_index=None, model_path=None, compile=True):
        if model_path is None and job_index is None:
            raise Exception("Must provide either a model path or a job index")

        model_path = model_path if model_path is not None else os.path.join(
            self.models_path, "model_job{}".format(job_index))

        if self.n_classes > 1:
            custom_objects = {
                "dice_loss": self.metric_funcs.dice_loss,
                "dice_coef_necrotic": self.metric_funcs.dice_coef_necrotic,
                "dice_coef_edema": self.metric_funcs.dice_coef_edema,
                "dice_coef_enhancing": self.metric_funcs.dice_coef_enhancing,
                "sensitivity": self.metric_funcs.sensitivity,
                "specificity": self.metric_funcs.specificity,
                "binary_cross_entropy_per_channel": self.metric_funcs.binary_cross_entropy_per_channel
            }
        else:
             custom_objects = {
                "sensitivity": self.metric_funcs.sensitivity,
                "specificity": self.metric_funcs.specificity,
            }

        if self.activation != 'softmax':
            custom_objects['combined_loss'] = self.metric_funcs.combined_loss

        self.model = load_model(
            model_path, custom_objects=custom_objects, compile=compile)

    # Starts or resumes training of the model for job's fraction of total n_epochs
    def train_model(self, job_index, n_jobs, n_epochs, batch_size = BATCH_SIZE):
        for i in range(job_index, n_jobs):
            job_index = i

            start_epoch = job_index * n_epochs // n_jobs
            end_epoch = (job_index + 1) * n_epochs // n_jobs

            # Load the model from a file
            model_filename = "/content/current/model_job{}".format(job_index)
            if start_epoch == 0:
                inputs = Input((IMG_SIZE, IMG_SIZE, self.n_channels))
                self.model = create_unet(
                    inputs, activation=self.activation, num_classes=self.n_classes, loss=self.metric_funcs.combined_loss, metrics=self.metrics)
            else:
                # Load previous model to continue training
                load_model(job_index-1)

        one_hot = self.activation == 'softmax'
        training_gen = create_training_gen(train_ids, one_hot=one_hot, slice_range=100, slice_start=22, slice_interval=5,
                                           modalities=self.modalities, batch_size=batch_size, dim=(IMG_SIZE, IMG_SIZE), n_classes=self.n_classes)
        test_gen = create_test_gen(test_ids, one_hot=one_hot, slice_range=100, slice_start=22, slice_interval=5,
                                   modalities=self.modalities, batch_size=batch_size, dim=(IMG_SIZE, IMG_SIZE), n_classes=self.n_classes)

        self.model.fit(training_gen,
                       epochs=end_epoch,
                       steps_per_epoch=len(train_ids)/batch_size,
                       initial_epoch=start_epoch,
                       callbacks=callbacks,
                       validation_data=test_gen)

        # Save the model to a file
        self.model.save(model_filename)

    # Evaluates the model on the test set
    def evaluate_model(self, batch_size=BATCH_SIZE):
        if self.model is None:
            print("Model not loaded")
        test_gen = create_test_gen(test_ids, slice_range=100, slice_start=22, slice_interval=5,
                                   modalities=self.modalities, batch_size=batch_size, dim=(IMG_SIZE, IMG_SIZE), n_classes=self.n_classes)
        self.model.evaluate(test_gen, steps=len(test_ids)/batch_size)

    # Predicts the segmentation of the given images
    def predict(self, images):
        X = np.empty(images.shape)
        for i in range(images.shape[0]):
            for chan in range(self.n_channels):
                X[i, ..., chan] = cv2.resize(
                    images[i, ..., chan], (IMG_SIZE, IMG_SIZE))

        X = X/np.max(X)
        return self.model.predict(X)
