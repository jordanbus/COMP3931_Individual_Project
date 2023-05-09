import pytest
from unet import create_unet
from tensorflow.keras.layers import Input
import tensorflow as tf

def test_unet_creates_model():
    inputs = Input((128, 128, 3))
    model = create_unet(inputs)
    
    assert isinstance(model, tf.keras.Model)
    
# def test_unet()