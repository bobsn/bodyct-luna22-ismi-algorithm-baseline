from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals
)

from math import ceil

import six
import tensorflow
from keras import backend as K
from keras.layers import BatchNormalization
from keras.layers import (
    Conv3D,
    AveragePooling3D,
    MaxPooling3D
)
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten
)
from keras.layers import add
from keras.models import Model
from keras.regularizers import l2


from typing import Dict

import SimpleITK
import numpy as np
from pathlib import Path
import json

import tensorflow.keras
from tensorflow.keras.applications import VGG16

# Enforce some Keras backend settings that we need
tensorflow.keras.backend.set_image_data_format("channels_first")
tensorflow.keras.backend.set_floatx("float32")
from data import (
    center_crop_volume,
    get_cross_slices_from_cube,
)


def clip_and_scale(
    data: np.ndarray,
    min_value: float = -1000.0,
    max_value: float = 400.0,
) -> np.ndarray:
    data = (data - min_value) / (max_value - min_value)
    data[data > 1] = 1.0
    data[data < 0] = 0.0
    return data


class Nodule_classifier:
    def __init__(self):

        self.input_size = 64
        self.input_spacing = 1.0

        self.model = build_model(input_shape=(1, 64, 64, 64), filter_list=[60, 66, 68, 70, 72])
        self.model.load_weights(
            "/opt/algorithm/models/dense_model_noduletype_best_val_accuracy.h5",
            by_name=True,
            skip_mismatch=True,
        )

        print("Models initialized")

    def load_image(self) -> SimpleITK.Image:

        ct_image_path = list(Path("/input/images/ct/").glob("*"))[0]
        image = SimpleITK.ReadImage(str(ct_image_path))

        return image

    def preprocess(
        self,
        img: SimpleITK.Image,
    ) -> SimpleITK.Image:

        # Resample image
        original_spacing_mm = img.GetSpacing()
        original_size = img.GetSize()
        new_spacing = (self.input_spacing, self.input_spacing, self.input_spacing)
        new_size = [
            int(round(osz * ospc / nspc))
            for osz, ospc, nspc in zip(
                original_size,
                original_spacing_mm,
                new_spacing,
            )
        ]
        resampled_img = SimpleITK.Resample(
            img,
            new_size,
            SimpleITK.Transform(),
            SimpleITK.sitkLinear,
            img.GetOrigin(),
            new_spacing,
            img.GetDirection(),
            0,
            img.GetPixelID(),
        )

        # Return image data as a numpy array
        return SimpleITK.GetArrayFromImage(resampled_img)

    def predict(self, input_image: SimpleITK.Image) -> Dict:

        print(f"Processing image of size: {input_image.GetSize()}")

        nodule_data = self.preprocess(input_image)

        # Crop a volume of 50 mm^3 around the nodule
        nodule_data = center_crop_volume(
            volume=nodule_data,
            crop_size=np.array(
                (
                    self.input_size,
                    self.input_size,
                    self.input_size,
                )
            ),
            pad_if_too_small=True,
            pad_value=-1024,
        )

        # Extract the axial/coronal/sagittal center slices of the 50 mm^3 cube
        # nodule_data = get_cross_slices_from_cube(volume=nodule_data)
        nodule_data = np.expand_dims(nodule_data, 0)
        nodule_data = clip_and_scale(nodule_data)

        # malignancy = self.model_malignancy(nodule_data[None]).numpy()[0, 1]
        # texture = np.argmax(self.model_nodule_type(nodule_data[None]).numpy())

        changed_array = np.expand_dims(nodule_data, 0)
        predictions = self.model(changed_array)
        malignancy = predictions[0].numpy()[0, 1]
        texture = np.argmax(predictions[1].numpy())


        result = dict(
            malignancy_risk=round(float(malignancy), 3),
            texture=int(texture),
        )

        return result

    def write_outputs(self, outputs: dict):

        with open("/output/lung-nodule-malignancy-risk.json", "w") as f:
            json.dump(outputs["malignancy_risk"], f)

        with open("/output/lung-nodule-type.json", "w") as f:
            json.dump(outputs["texture"], f)

    def process(self):

        image = self.load_image()
        result = self.predict(image)
        self.write_outputs(result)
"""
Dense network
"""
def classification_layer(inputs):
    x = tensorflow.keras.layers.Dense(512, activation='relu')(inputs)
    x = tensorflow.keras.layers.Dense(256, activation='relu')(x)
    output_type = tensorflow.keras.layers.Dense(3, activation='softmax', name='type_classification')(x)
    return output_type


def regression_layer(inputs):
    x = tensorflow.keras.layers.Dense(512, activation='relu')(inputs)
    x = tensorflow.keras.layers.Dense(256, activation='relu')(x)
    output_mal = tensorflow.keras.layers.Dense(2, activation='softmax', name='malignancy_regression')(x)
    return output_mal


def add_dense_blocks_old(inputs, filter_size):
    for i in range(3):
        x_1 = tensorflow.keras.layers.Conv3D(filters=filter_size, kernel_size=(1, 1, 1), padding='same',
                                             activation='relu', kernel_initializer='he_normal')(inputs)
        if i != 0:
            inputs = tensorflow.keras.layers.Concatenate(axis=1)([x_1, inputs])  # axis=1
        else:
            inputs = x_1
        x_2 = tensorflow.keras.layers.Conv3D(filters=filter_size, kernel_size=(3, 3, 3), padding='same',
                                             activation='relu', kernel_initializer='he_normal')(inputs)
        inputs = tensorflow.keras.layers.Concatenate(axis=1)([x_2, inputs])  # axis=1
        filter_size += 32
    return inputs


def add_dense_blocks(inputs, filter_size):
    x_1 = tensorflow.keras.layers.Conv3D(filters=filter_size, kernel_size=(1, 1, 1), padding='same',
                                         activation='relu', kernel_initializer='he_normal')(inputs)
    x_2 = tensorflow.keras.layers.Conv3D(filters=filter_size, kernel_size=(3, 3, 3), padding='same',
                                         activation='relu', kernel_initializer='he_normal')(x_1)
    concat_x1 = x_1
    concat_x2 = x_2
    filter_size += 8  # 32
    for i in range(2):
        x_1 = tensorflow.keras.layers.Conv3D(filters=filter_size, kernel_size=(1, 1, 1), padding='same',
                                             activation='relu', kernel_initializer='he_normal',
                                             kernel_regularizer='L2')(concat_x2)
        concat_x1 = tensorflow.keras.layers.Concatenate(axis=1)([x_1, concat_x1])  # axis=1
        x_2 = tensorflow.keras.layers.Conv3D(filters=filter_size, kernel_size=(3, 3, 3), padding='same',
                                             activation='relu', kernel_initializer='he_normal',
                                             kernel_regularizer='L2')(concat_x1)
        concat_x2 = tensorflow.keras.layers.Concatenate(axis=1)([x_2, concat_x2])  # axis=1
        filter_size += 16  # 32
    return x_2


def add_transition_blocks(x, transition_filter):
    x = tensorflow.keras.layers.Conv3D(filters=transition_filter, kernel_size=(1, 1, 1), padding='same',
                                       activation='relu', kernel_initializer='he_normal')(x)
    x = tensorflow.keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(x)
    return x


def add_network_blocks(x, block_amount, filter_list):
    # Create dense block
    for block in range(block_amount):
        # Add dense block
        x = add_dense_blocks(x, filter_list[block])
        # Create transition layer
        x = add_transition_blocks(x, filter_list[block]//2)
    return x


def build_model(input_shape, filter_list=[160, 176, 184, 188, 190]):
    input_layer = tensorflow.keras.layers.Input(shape=input_shape)
    x = tensorflow.keras.layers.Conv3D(filters=64, kernel_size=(3, 3, 3), padding='same',
                                       activation='relu', kernel_initializer='he_normal')(input_layer)
    x = add_network_blocks(x, 4, filter_list)
    x = tensorflow.keras.layers.MaxPooling3D()(x)

    x = add_dense_blocks(x, filter_list[-1])

    x = tensorflow.keras.layers.AveragePooling3D((2, 2, 2))(x)
    x = tensorflow.keras.layers.Flatten(name='bifurcation_layer')(x)
    output_malignancy = regression_layer(x)
    output_type = classification_layer(x)
    d_model = tensorflow.keras.Model(inputs=input_layer, outputs=[output_malignancy, output_type])
    return d_model


def _handle_data_format():
    global DIM1_AXIS
    global DIM2_AXIS
    global DIM3_AXIS
    global CHANNEL_AXIS
    if K.image_data_format() == 'channels_last':
        DIM1_AXIS = 1
        DIM2_AXIS = 2
        DIM3_AXIS = 3
        CHANNEL_AXIS = 4
    else:
        CHANNEL_AXIS = 1
        DIM1_AXIS = 2
        DIM2_AXIS = 3
        DIM3_AXIS = 4


def _get_block(identifier):
    if isinstance(identifier, six.string_types):
        res = globals().get(identifier)
        if not res:
            raise ValueError('Invalid {}'.format(identifier))
        return res
    return identifier


if __name__ == "__main__":
    Nodule_classifier().process()
