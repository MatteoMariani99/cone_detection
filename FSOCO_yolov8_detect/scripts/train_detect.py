import comet_ml
from ultralytics import YOLO
from torchinfo import summary
import torch
import os

# Get the value of $HOME
home_dir = os.environ.get('HOME')

# nel terminale
# export COMET_API_KEY=umDmt9f9wcCwCg1IqdPDXHX9G

#! visualizzazione su comet -> fare il login e tutto si vede in real time
#comet_ml.init(project_name="comet-example-yolov8-FSOCO")
torch.cuda.set_device(0)

# modello yolo
yoloModel = "yolov8s.pt"
model = YOLO(yoloModel,'gpu')

# Controllo che il modello selezionato sia quello corretto
summary(model)

# batch=-1 regola dinamicamente la dimensione in base alla quantita di memoria disponbile nella GPU
# optimizer = 'auto'
results = model.train(
    data = home_dir+'/cone_detection/fsoco_dataset_yolov8/data.yaml',
    imgsz = 640,
    epochs = 10,
    batch = -1,
    optimizer = 'auto',
    name = 'prova',
    #project = 'comet-example-yolov8-FSOCO'
    project = '../result_detect_train'
)
