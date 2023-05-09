import pytest
from metrics import MulticlassMetrics, BinaryMetrics
from unet_model import UNetModel
import numpy as np

def test_constructor_initialises_multilabel_correctly():
    
    model = UNetModel(
        models_path="/content/models", 
        loss='combined_loss',
        modalities=['flair', 't1ce', 't2'], 
        slice_interval=50,
        dummy_ids=True
        
    )

    # Loss should be a function, not a string
    assert callable(model.loss)
    
    assert model.n_classes == 4
    assert model.activation == 'sigmoid'
    metrics = ['accuracy',
            MulticlassMetrics.sensitivity,
            MulticlassMetrics.specificity,
            MulticlassMetrics.binary_cross_entropy_per_channel_wrapper(model.class_weights),
            MulticlassMetrics.dice_loss_wrapper(model.class_weights),
            MulticlassMetrics.dice_coef_necrotic,
            MulticlassMetrics.dice_coef_edema,
            MulticlassMetrics.dice_coef_enhancing,
            MulticlassMetrics.combined_loss_wrapper(0.5)]
    assert len(metrics) == len(model.metrics)
    assert model.model is None
    
def test_constructor_initialises_binary_correctly():
    
    model = UNetModel(
        models_path="/content/models", 
        loss='combined_loss',
        segment_classes={1:'tumor'},
        modalities=['flair', 't1ce', 't2'], 
        slice_interval=50,
        dummy_ids=True
    )

    # Loss should be a function, not a string
    assert callable(model.loss)
    
    assert model.n_classes == 1
    assert model.activation == 'sigmoid'
    metrics = ['accuracy',
                BinaryMetrics.sensitivity,
                BinaryMetrics.specificity,
                BinaryMetrics.dice_loss,
                BinaryMetrics.combined_loss_wrapper()]
    assert len(metrics) == len(model.metrics)
    assert model.model is None
    
def test_constructor_raises_error_for_invalid_loss():
    with pytest.raises(ValueError):
        model = UNetModel(
            models_path="/content/models", 
            loss='invalid_loss', 
            modalities=['flair', 't1ce', 't2'], 
            slice_interval=50,
        dummy_ids=True
            
        )
        
def test_constructor_raises_error_for_invalid_modalities():
    with pytest.raises(ValueError):
        model = UNetModel(
            models_path="/content/models", 
            loss='combined_loss', 
            modalities=['flair', 't1ce', 'invalid_modality'], 
            slice_interval=50,
            dummy_ids=True
        )
        
        
def test_constructor_gets_train_ids():
    model = UNetModel(
        models_path="/content/models", 
        loss='combined_loss', 
        modalities=['flair', 't1ce', 't2'], 
        slice_interval=50,
        dummy_ids=True
        
    )
    
    assert len(model.train_ids) > 0
    
def test_constructor_gets_test_ids():
    model = UNetModel(
        models_path="/content/models", 
        loss='combined_loss', 
        modalities=['flair', 't1ce', 't2'], 
        slice_interval=50,
        dummy_ids=True
        
    )
    
    assert len(model.test_ids) > 0
    
def test_compile_model_raises_exception_when_no_model_loaded():
    model = UNetModel(
        models_path="/content/models", 
        loss='combined_loss', 
        modalities=['flair', 't1ce', 't2'], 
        slice_interval=50,
        dummy_ids=True
        
    )
    
    with pytest.raises(Exception):
        model.compile_model()
    
    
def test_train_model_raises_exception_when_no_model_loaded():
    model = UNetModel(
        models_path="/content/models", 
        loss='combined_loss', 
        modalities=['flair', 't1ce', 't2'], 
        slice_interval=50,
        dummy_ids=True
        
    )
    
    with pytest.raises(Exception):
        model.train_model()
        
def test_evaluate_model_raises_exception_when_no_model_loaded():
    model = UNetModel(
        models_path="/content/models", 
        loss='combined_loss', 
        modalities=['flair', 't1ce', 't2'], 
        slice_interval=50,
        dummy_ids=True
        
    )
    
    with pytest.raises(Exception):
        model.evaluate_model()
        
def test_predict_raises_exception_when_no_images_provided():
    model = UNetModel(
        models_path="/content/models", 
        loss='combined_loss', 
        modalities=['flair', 't1ce', 't2'], 
        slice_interval=50,
        dummy_ids=True

    )
    
    # Set model to a string to simulate a loaded model
    model.model = 'test'
    
    with pytest.raises(Exception):
        model.predict([])