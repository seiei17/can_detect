# Can_detect
For can detection

## Inference module

file: inference/inference.py: inference().

```python
import inference/inference.Inference

# use cpu, devices='cpu' or cuda devices='0' or '0,1,2,3'
inf = Inference(weight='./weights/best.pt',
                imgsz=640, 
                devices='0', 
                conf_thres=.4, 
                iou_thres=.5)

# res -> (class, x1, y1, x2, y2)
res = inf.inference(img0)
```
