# Can_detect
For can detection

## Project Framework

models/: model files

utils/: inference utils

weight/: weight files

inference.py: inference file

## How to start?

### install dependencies

```python
pip install -U -r requirement.txt
```

### Inference module

file: inference.py: Inference().

```python
from inference import Inference

# use cpu, devices='cpu' or cuda devices='0' or '0,1,2,3'
inf = Inference(weight='./weights/best.pt',
                imgsz=640, 
                devices='0', 
                conf_thres=.4, 
                iou_thres=.5)

# res -> (class, x1, y1, x2, y2)
res = inf.inference(img0)
```
