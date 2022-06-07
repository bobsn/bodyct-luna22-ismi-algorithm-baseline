from typing import Dict

import SimpleITK
import numpy as np
from pathlib import Path
import json

import tensorflow.keras
from tensorflow.keras.applications import VGG16
from resnet_3d import CustomResnet3DBuilder

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

        self.model = CustomResnet3DBuilder.build_resnet_18((1, 64, 64, 64), 3)
        self.model.load_weights(
            "/opt/algorithm/models/resnet18.h5",
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
        predictions = self.model(nodule_data[None]).numpy()
        malignancy = predictions[0][0, 1]
        texture = predictions[1]

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


if __name__ == "__main__":
    Nodule_classifier().process()
