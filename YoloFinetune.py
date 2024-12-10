# YOlO Finetune
# source of YOLO V8: https://github.com/AlexeyAB/darknet
from ultralytics import YOLO
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load Model 
model = YOLO("yolo11n.pt") # load a pretrained model (recommended for training)
# Train the model with MPS 
results = model.train(
    data = "data.yaml", 
    epochs =20, 
    imgsz = 640, 
    device = "cpu")
