import os
from functools import partial

import pydicom
import numpy as np
import tensorflow as tf
from skimage.transform import resize


def get_dicom_data(patients_root, n_depth=5, rows=64, columns=64):
    def gen(patients_root):
        for patient_dir in os.listdir(patients_root):
            patient_dir = os.path.join(patients_root, patient_dir)
            patient_files = [os.path.join(e) for e in os.listdir(patient_dir)]
            patient_files.sort(key=lambda fname: int(fname.split('.')[0]))
            dcm_slices = [pydicom.read_file(os.path.join(patient_dir, f)) for f in patient_files]

            # Resample slices such that the depth of the CT scan is 'n_depth'
            slice_group = n_depth / len(patient_files)
            slice_indexes = [int(idx / slice_group) for idx in range(n_depth)]
            dcm_slices = [dcm_slices[i] for i in slice_indexes]

            # Merge slices
            shape = (rows, columns)
            shape = (n_depth, *shape)
            img = np.empty(shape, dtype='float32')
            for idx, dcm in enumerate(dcm_slices):
                # Rescale and shift in order to get accurate pixel values
                slope = float(dcm.RescaleSlope)
                intercept = float(dcm.RescaleIntercept)
                resized_img = resize(dcm.pixel_array.astype('float32'), (rows, columns), anti_aliasing=True)
                img[idx, ...] = resized_img * slope + intercept
            yield img

    return tf.data.Dataset.from_generator(partial(gen, patients_root), output_types=tf.float32)
