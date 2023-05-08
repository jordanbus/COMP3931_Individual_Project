from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, BatchNormalization, ReLU, Concatenate, Conv2DTranspose


# Architecture inspired by DigitalSrini's U-Net tutorial at https://youtu.be/68HR_eyzk00
# Creates a U-Net model with the given parameters, using keras Layers
def create_unet(inputs, loss="categorical_crossentropy", activation='sigmoid', num_filters=32, num_classes=1, kernel_initializer='he_normal', dropout_rate=0.2, metrics=['accuracy'], compile=True):

    # Contracting path of U-Net
    conv1, pool1 = encode_conv(inputs, num_filters, dropout_rate=dropout_rate, kernel_initializer=kernel_initializer)
    conv2, pool2 = encode_conv(pool1, num_filters*2, dropout_rate=dropout_rate, kernel_initializer=kernel_initializer)
    conv3, pool3 = encode_conv(pool2, num_filters*4, dropout_rate=dropout_rate, kernel_initializer=kernel_initializer)
    conv4, pool4 = encode_conv(pool3, num_filters*8, dropout_rate=dropout_rate, kernel_initializer=kernel_initializer)
    # Bottleneck
    conv5 = bottleneck_conv(pool4, num_filters*16, dropout_rate=dropout_rate, kernel_initializer=kernel_initializer)
    # Expansive path
    up6 = decode_conv(conv5, conv4, num_filters*8, dropout_rate=dropout_rate, kernel_initializer=kernel_initializer)
    up7 = decode_conv(up6, conv3, num_filters*4, dropout_rate=dropout_rate, kernel_initializer=kernel_initializer)
    up8 = decode_conv(up7, conv2, num_filters*2, dropout_rate=dropout_rate, kernel_initializer=kernel_initializer)
    up9 = decode_conv(up8, conv1, num_filters, concat_axis=3, dropout_rate=dropout_rate, kernel_initializer=kernel_initializer)

    # Get final convolutional layer with channels for each class only
    # Using activation function to get probabilities
    outputs = Conv2D(num_classes, (1,1), activation=activation)(up9)

    # Create a Keras model using the input and output layers
    model = Model(inputs=inputs, outputs=outputs)
    
    if compile:
        model.compile(loss=loss, optimizer=Adam(learning_rate=1e-3), metrics = metrics)

    return model

# Encodes one layer of the contracting path of the U-Net
def encode_conv(inputs, filters, dropout_rate=0.2, kernel_initializer='he_normal'):
    conv = Conv2D(filters, (3,3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(inputs)
    conv = BatchNormalization()(conv)
    conv = Dropout(dropout_rate)(conv)
    conv = Conv2D(filters, (3,3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(conv)
    conv = BatchNormalization()(conv)
    pool = MaxPooling2D(pool_size=(2, 2))(conv)
    return conv,pool

# Bottleneck layer of the U-Net
def bottleneck_conv(inputs, filters, dropout_rate=0.2, kernel_initializer='he_normal'):
    conv = Conv2D(filters, (3,3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(inputs)
    conv = BatchNormalization()(conv)
    conv = Dropout(dropout_rate)(conv)
    conv = Conv2D(filters, (3,3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(conv)
    conv = BatchNormalization()(conv)
    return conv

# Decodes one layer of the expansive path of the U-Net
def decode_conv(inputs, skip_connection, filters, concat_axis=-1, dropout_rate=0.2, kernel_initializer='he_normal'):
    up = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(inputs)
    up = Concatenate(axis=concat_axis)([up, skip_connection])
    conv = Conv2D(filters, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(up)
    conv = BatchNormalization()(conv)
    conv = Dropout(dropout_rate)(conv)
    conv = Conv2D(filters, (3, 3), activation='relu', kernel_initializer=kernel_initializer, padding='same')(conv)
    conv = BatchNormalization()(conv)
    return conv