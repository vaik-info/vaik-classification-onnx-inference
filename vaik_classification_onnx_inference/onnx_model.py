from typing import List, Dict, Tuple
import onnxruntime as rt
from PIL import Image
import numpy as np


class OnnxModel:
    def __init__(self, input_saved_model_path: str = None, classes: Tuple = None):
        self.model, self.mean, self.std = self.__load(input_saved_model_path)
        self.model_input_shape = tuple(self.model.get_inputs()[0].shape)
        self.input_name = self.model.get_inputs()[0].name
        self.model_output_shape = tuple(self.model.get_outputs()[0].shape)
        self.output_name = self.model.get_outputs()[0].name
        self.classes = classes

    def __load(self, input_saved_model_path):
        model = rt.InferenceSession(input_saved_model_path, providers=rt.get_available_providers())
        meta = model.get_modelmeta()
        mean = np.array(eval(meta.custom_metadata_map['normalize_mean'])).reshape(-1, 1, 1)
        std = np.array(eval(meta.custom_metadata_map['normalize_std'])).reshape(-1, 1, 1)
        return model, mean, std

    def inference(self, input_image_list: List[np.ndarray], batch_size: int = 8) -> Tuple[List[Dict], Dict]:
        resized_image_array = self.__preprocess_image_list(input_image_list, self.model_input_shape[2:4])
        raw_pred = self.__inference(resized_image_array, batch_size)
        output = self.__output_parse(raw_pred)
        return output, raw_pred

    def __inference(self, resize_input_tensor: np.ndarray, batch_size: int) -> np.ndarray:
        if len(resize_input_tensor.shape) != 4:
            raise ValueError('dimension mismatch')

        output_tensor = np.zeros((resize_input_tensor.shape[0], self.model_output_shape[-1]))
        for index in range(0, resize_input_tensor.shape[0], batch_size):
            batch = resize_input_tensor[index:index + batch_size, :, :, :]
            batch_pad = np.zeros(((batch_size, ) + self.model_input_shape[2:4] + (self.model_input_shape[1], )), dtype=np.float32)
            batch_pad[:batch.shape[0], :, :, :] = batch
            batch_pad_normalized = self.__normalize(batch_pad)
            raw_pred = self.model.run([self.output_name], {self.input_name: batch_pad_normalized})
            output_tensor[index:index + batch.shape[0], :] = np.stack(raw_pred[:batch.shape[0]], axis=0)
        return output_tensor

    def __preprocess_image_list(self, input_image_list: List[np.ndarray],
                                resize_input_shape: Tuple[int, int]) -> np.ndarray:
        resized_image_list = []
        for input_image in input_image_list:
            resized_image = self.__preprocess_image(input_image, resize_input_shape)
            resized_image_list.append(resized_image)
        return np.stack(resized_image_list)

    def __preprocess_image(self, input_image: np.ndarray, resize_input_shape: Tuple[int, int]) -> Tuple[
        np.ndarray, Tuple[float, float]]:
        if len(input_image.shape) != 3:
            raise ValueError('dimension mismatch')
        if not np.issubdtype(input_image.dtype, np.uint8):
            raise ValueError(f'dtype mismatch expected: {np.uint8}, actual: {input_image.dtype}')

        output_image = np.zeros((*resize_input_shape, input_image.shape[2]),
                                dtype=input_image.dtype)
        pil_image = Image.fromarray(input_image)
        x_ratio, y_ratio = resize_input_shape[1] / pil_image.width, resize_input_shape[0] / pil_image.height
        if x_ratio < y_ratio:
            resize_size = (resize_input_shape[1], round(pil_image.height * x_ratio))
        else:
            resize_size = (round(pil_image.width * y_ratio), resize_input_shape[0])
        resize_pil_image = pil_image.resize(resize_size)
        resize_image = np.array(resize_pil_image)
        output_image[:resize_image.shape[0], :resize_image.shape[1], :] = resize_image
        return output_image

    def __normalize(self, batch_pad):
        batch_pad = np.transpose(batch_pad, [0, 3, 1, 2]) / 255.
        batch_pad_normalized = (batch_pad - self.mean) / self.std
        return batch_pad_normalized.astype(np.float32)

    def __output_parse(self, pred: np.ndarray) -> List[Dict]:
        output_dict_list = []
        pred_index = np.argsort(-pred, axis=-1)
        for index in range(pred.shape[0]):
            output_dict = {'score': pred[index][pred_index[index]].tolist(),
                           'label': [self.classes[class_index] for class_index in pred_index[index]]}
            output_dict_list.append(output_dict)
        return output_dict_list
