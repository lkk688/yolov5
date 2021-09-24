import torch

# Model
#model = torch.hub.load('ultralytics/yolov5', 'yolov5m')  # yolov5s or yolov5m, yolov5l, yolov5x, custom
#Downloading https://github.com/ultralytics/yolov5/releases/download/v5.0/yolov5s.pt to /home/lkk/.cache/torch/hub/ultralytics_yolov5_master/yolov5s.pt...

model = torch.hub.load('ultralytics/yolov3', 'yolov3')  # or 'yolov3_spp', 'yolov3_tiny'
# Images
img = 'https://ultralytics.com/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list

# Inference
results = model(img)

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
