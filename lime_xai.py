from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from unet_model import UNetModel, IMG_SIZE
from dataset import create_test_gen, get_train_test_ids_completed
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed


class LimeXAI:
    def __init__(self, model, seed=None):
        # Model should be of class UNetModel
        if not isinstance(model, UNetModel):
            raise Exception("Model must be of type UNetModel")
        self.model = model
        self.explainer = lime_image.LimeImageExplainer(random_state=seed)
        train_ids, test_ids = get_train_test_ids_completed(
            mri_types=model.modalities)
        self.test_gen = create_test_gen(test_ids, one_hot=model.loss == 'categorical_crossentropy', slice_range=self.model.slice_range,
                                   slice_start=self.model.slice_start, slice_interval=self.model.slice_interval,
                                   modalities=self.model.modalities, batch_size=2, dim=(IMG_SIZE, IMG_SIZE),
                                   segment_classes=self.model.segment_classes, seed=seed, shuffle=False)
        self.X_test, self.y_test = self.test_gen.__getitem__(0)
        nsamples, nx, ny, nc = self.X_test.shape
        self.X_test_reshaped = self.X_test.reshape((nsamples, nx*ny*nc))

    # Get a new batch of images and masks from test generator
    def get_batch(self, index):
        self.X_test, self.y_test = self.test_gen.__getitem__(index)

    # # Define a predict function that outputs image mask prediction
    def predict_fn(self, images):
        # remove any extra channels that were added for lime
        reverted_images = np.stack(
            (images[..., i] for i in range(self.model.n_channels)), axis=-1)

        masks = self.model.predict(reverted_images)
        # masks = (masks > 0.5).astype(int)

        nsamples, nx, ny, nc = masks.shape
        return masks.reshape((nsamples, nx*ny*nc))

    # Combine all tumour classes into one for prediction mask
    def predict_whole_fn(self, images):
        # remove any extra channels that were added for lime
        reverted_images = np.stack(
            (images[..., i] for i in range(self.model.n_channels)), axis=-1)

        masks = self.model.predict(reverted_images)
        # Threshold masks to 0 or 1
        masks = (masks > 0.5).astype(int)
        # Combine all tumour classes into one mask
        mask = masks[..., 1] | masks[..., 2] | masks[..., 3]
        nsamples, nx, ny = mask.shape
        return mask.reshape((nsamples, nx*ny))

    def explain(self, sample, num_samples=100, visualize=False, duplicate_channel=1, segmentation_fn=None):
        image = self.X_test[sample]
        # Lime expects 3 channels, so duplicate channel for other channels
        if image.shape[-1] == 2:
            if str(duplicate_channel) == 'mean':
                dup_image = (image[..., 0] + image[..., 1]) / 2
            else:
                dup_image = image[..., duplicate_channel]
            
            image = np.stack(
                (image[..., 0], image[..., 1], dup_image), axis=-1)
        elif image.shape[-1] == 1:
            image = np.stack(
                (image[..., 0], image[..., 0], image[..., 0]), axis=-1)

        explanation = self.explainer.explain_instance(
            image,
            self.predict_fn,
            top_labels=self.model.n_classes,
            segmentation_fn=segmentation_fn,
            #  hide_color=0,
            num_samples=num_samples,
        )

        if visualize:
            self._visualize_explanation(sample, explanation)

        return explanation

    def _visualize_explanation(self, sample, explanation):
        gt_colors = np.array([[0, 0, 0, 0], [255, 0, 0, 100], 
                              [0, 255, 0, 100], [0, 0, 255, 100]])
        pred_colors = np.array([[0,0,0,0],[0, 255, 0, 100], 
                               [0, 0, 255, 100], [255, 0, 0, 100]] )

        image = self.X_test[sample]
        ground_truth = self.y_test[sample]
        predictions = self.model.model.predict(self.X_test)
        prediction = predictions[sample]

        for i in range(self.model.n_classes):
                

            # Get the image and mask from the explanation
            img, mask = explanation.get_image_and_mask(
                explanation.top_labels[i],
                positive_only=True,
                hide_rest=True,
            )
           

            
            pred = (prediction[..., i])
            
           # For binary classification, only show the positive class
            if self.model.n_classes == 1:
                gt = ground_truth
                i = 1
            else:
                gt = ground_truth[..., i]
                
            
            gt_sm = np.zeros(
                (ground_truth.shape[0], ground_truth.shape[1], 4), dtype=np.uint8)
            pred_sm = np.zeros(
                (ground_truth.shape[0], ground_truth.shape[1], 4), dtype=np.uint8)
            
            gt_sm[gt == 1] = gt_colors[i]
            pred_sm[pred >= 0.8] = pred_colors[i]

            f, axarr = plt.subplots(1, self.model.n_channels * 3 + 1, figsize=(18, 50))
            plt.figure(figsize=(18, 50))
            for j in range(self.model.n_channels):  
                # Overlay the binary mask on the original images
                axarr[j*3].imshow(image[...,j], cmap='gray')
                axarr[j*3].imshow(gt_sm, cmap='jet')
                axarr[j*3].set_title('{} with GT'.format(self.model.modalities[j]))

                axarr[j*3+1].imshow(image[...,j], cmap='gray')
                axarr[j*3+1].imshow(pred_sm, cmap='jet')
                # axarr[j*3+1].imshow(mask, cmap='jet', alpha=0.5)
                axarr[j*3+1].set_title('{} with Pred'.format(self.model.modalities[j]))
                
                axarr[j*3+2].imshow(image[...,j], cmap='gray')
                axarr[j*3+2].imshow(pred_sm, cmap='jet')
                axarr[j*3+2].imshow(mask, cmap='jet', alpha=0.5)
                axarr[j*3+2].set_title('{} with Pred + mask'.format(self.model.modalities[j]))

                # axarr[j*2+1].imshow(mark_boundaries(image[...,j], mask))
                # axarr[j*2+1].set_title('Image + LIME')

            axarr[-1].imshow(gt_sm, cmap='gray')
            axarr[-1].imshow(pred_sm, cmap='jet')
            axarr[-1].set_title('GT vs Prediction')
            # axarr[0].imshow(mask, cmap='jet', alpha=0.5)
            # axarr[1].imshow(mask, cmap='jet', alpha=0.5)
            # axarr[3].imshow(mask, cmap='jet', alpha=0.5)
            # if segment_classes is not None:
            print(self.model.segment_classes[i])

        # plt.axis('off')
        plt.show()
