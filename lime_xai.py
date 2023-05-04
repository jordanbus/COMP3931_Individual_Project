import lime
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
from train_model import UNetModel, MODALITIES
from dataset import create_test_gen, get_train_test_ids_completed
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed

    
    
class LimeXAI:
    def __init__(self, model):
        # Model should be of class UNetModel
        if not isinstance(model, UNetModel):
            raise Exception("Model must be of type UNetModel")
        self.model = model
        self.explainer = lime_image.LimeImageExplainer(random_state=42)
        train_ids, test_ids = get_train_test_ids_completed(mri_types=MODALITIES)
        test_gen = create_test_gen(test_ids, slice_range=100, slice_start=22, slice_interval=5, modalities = MODALITIES, batch_size = 1, dim = (128, 128), n_classes = model.n_classes)
        self.X_test, self.y_test = test_gen.__getitem__(0)
        nsamples, nx, ny, nc = self.X_test.shape
        self.X_test_reshaped = self.X_test.reshape((nsamples,nx*ny*nc))
        
    # # Define a predict function that outputs image mask prediction
    def predict_fn(self,images):
        # remove any extra channels that were added for lime
        reverted_images = np.stack((images[...,i] for i in range(self.model.n_channels)), axis=-1)

        masks = self.model.predict(reverted_images)

        nsamples, nx, ny, nc = masks.shape
        return masks.reshape((nsamples,nx*ny*nc))
    
    # Combine all tumour classes into one for prediction mask
    def predict_whole_fn(self,images):
        # remove any extra channels that were added for lime
        reverted_images = np.stack((images[...,i] for i in range(self.model.n_channels)), axis=-1)

        masks = self.model.predict(reverted_images)
        masks = (masks>0.5).astype(int)
        mask = masks[...,1] | masks[...,2] | masks[...,3]
        nsamples, nx, ny = mask.shape
        return mask.reshape((nsamples,nx*ny))
    
    def explain(self, sample, num_samples=100, visualize=False, segmentation_fn=None, segment_classes=None, seed=None):
        image = self.X_test[sample]
        if image.shape[-1] == 2:
            image = np.stack((image[...,0],image[...,1],image[...,0]), axis=-1) 
        elif image.shape[-1] == 1:
            image = np.stack((image[...,0],image[...,0],image[...,0]), axis=-1) 
            
        explanation = self.explainer.explain_instance(
                image,
                self.predict_fn,
                top_labels=self.model.n_classes,
                segmentation_fn=felzenszwalb,
                #  hide_color=0,
                num_samples=num_samples,
                random_state=seed,
        )
        
        if visualize:
            self._visualize_explanation(sample, explanation, segment_classes=segment_classes)
            
        return explanation
    
    def _visualize_explanation(self, sample, explanation, segment_classes=None):
        gt_colors = np.array([[0, 0, 0, 0], [255, 0, 0, 100], [0, 255, 0, 100], [0, 0, 255, 100]])
        pred_colors = np.array([[0, 0, 0, 0], [122, 200, 0, 100], [34, 56, 40, 100], [0, 43, 150, 100]])
        
        image = self.X_test[sample]
        ground_truth = self.y_test[sample]
        predictions = self.model.model.predict(self.X_test)
        prediction = predictions[sample]

        gt_sm_whole = np.zeros((ground_truth.shape[0], ground_truth.shape[1], 4), dtype=np.uint8)
        pred_sm_whole = np.zeros((ground_truth.shape[0], ground_truth.shape[1], 4), dtype=np.uint8)

        for i in range(5):
            gt_sm = np.zeros((ground_truth.shape[0], ground_truth.shape[1], 4), dtype=np.uint8)
            pred_sm = np.zeros((ground_truth.shape[0], ground_truth.shape[1], 4), dtype=np.uint8)
            if i < 4:
                gt = (ground_truth[..., i])
                gt_sm[gt == 1] = gt_colors[i]
                gt_sm_whole[gt == 1] = gt_colors[i]
                pred = (prediction[..., i])
                pred_sm[pred >= 0.5] = pred_colors[i]
                pred_sm_whole[pred >= 0.5] = pred_colors[i]

                # Get the image and mask from the explanation
                img, mask = explanation.get_image_and_mask(
                        explanation.top_labels[i],
                        positive_only=True,
                        hide_rest=True,
                )
            else:
                gt_sm = gt_sm_whole
                pred_sm = pred_sm_whole

            # Get the original images
            t1ce = image[..., 0]
            flair = image[..., 1]
            
            # Overlay the binary mask on the original images
            plt.figure(figsize=(18, 50))
            f, axarr = plt.subplots(1, 6, figsize=(18, 50))
            axarr[0].imshow(t1ce, cmap='gray')
            axarr[0].imshow(gt_sm, cmap='jet')
            axarr[0].set_title('T1ce with GT')


            axarr[1].imshow(t1ce, cmap='gray')
            axarr[1].imshow(pred_sm, cmap='jet')
            axarr[1].set_title('T1ce with Pred')
            
            axarr[2].imshow(flair, cmap='gray')
            axarr[2].imshow(gt_sm, cmap='jet')
            axarr[2].set_title('Flair with GT')

            axarr[3].imshow(flair, cmap='gray')
            axarr[3].imshow(pred_sm, cmap='jet')
            axarr[3].set_title('Flair with Pred')

            # axarr[4].imshow(pred_sm, cmap='gray')
            # axarr[4].set_title('Prediction w/ Mask')
            axarr[4].imshow(mark_boundaries(flair, mask))
            axarr[4].set_title('Image + LIME')

            # axarr[5].imshow(t1ce, cmap='gray')
            axarr[5].imshow(gt_sm, cmap='gray')
            axarr[5].imshow(pred_sm, cmap='jet')
            axarr[5].set_title('GT vs Prediction')
            if i < 4:
                axarr[0].imshow(mask, cmap='jet', alpha=0.5)
                axarr[1].imshow(mask, cmap='jet', alpha=0.5)
                axarr[2].imshow(mask, alpha=0.5)
                # if segment_classes is not None:
                    # print(segment_classes[i])
            else:
                print("WHOLE Tumour")
            
        # plt.axis('off')
        if segment_classes is not None:
            plt.title(segment_classes[i])
        plt.show()

    
    