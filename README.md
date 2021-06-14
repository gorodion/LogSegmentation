# LogSegmentation
We have photos of timber trucks and we need to segment logs

Model, disparity maps & data: https://drive.google.com/drive/folders/18A9Aoew0tTzEYwugS-cCLKW3v1or7pyH?usp=sharing
## Instruction
Firstly specify PATH in config.py as path to model & disparity maps. Next:
```python
from inference import inference_model
import cv2

img = cv2.imread('some_img_8006.jpg')
mask = inference_model(img, 8006)
```
