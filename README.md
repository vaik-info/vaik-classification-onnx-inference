# vaik-classification-onnx-inference

Inference by classification ONNX model


## Install

``` shell
pip install git+https://github.com/vaik-info/vaik-classification-onnx-inference.git
```

## Requirement
- onnx model input shape
  - (batch, height, width, channel)

## Usage

### Example

```python
import os
import numpy as np
from PIL import Image

from vaik_classification_onnx_inference.onnx_model import OnnxModel

input_saved_model_dir_path = os.path.expanduser('~/.vaik-classification-pth-trainer/output_model/onnx/model.onnx')
classes = ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine')
image = np.asarray(
    Image.open(os.path.expanduser('~/.vaik-mnist-classification-dataset/valid/eight/valid_000000013.jpg')).convert('RGB'))

model = OnnxModel(input_saved_model_dir_path, classes)
output, raw_pred = model.inference([image], batch_size=1)
```

#### Output

- output

```text
[
  {
    'score': array(
  [
    1.0000000e+00,
    3.1797402e-11,
    2.7306603e-11,
    1.9686072e-11,
    4.6824385e-12,
    3.6455882e-12,
    8.8431291e-13,
    4.4509324e-13,
    4.5829687e-14,
    7.6756319e-17
  ],
  dtype=float32),
  'label'
  :
  [
    'eight',
    'nine',
    'two',
    'zero',
    'three',
    'four',
    'five',
    'six',
    'seven',
    'one'
  ]
  },
  ・・・
  ,
    'one'
  ]
  }
]
```

- raw_pred
```
[
  [
    1.9686072e-11
    7.6756319e-17
    2.7306603e-11
    4.6824385e-12
    3.6455882e-12
    8.8431291e-13
    4.4509324e-13
    4.5829687e-14
    1.0000000e+00
    3.1797402e-11
  ],
・・・
    3.1797402e-11
  ]
]
```