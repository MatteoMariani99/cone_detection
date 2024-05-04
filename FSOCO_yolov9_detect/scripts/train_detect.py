import comet_ml
from ultralytics import YOLO
from torchinfo import summary
import torch


# nel terminale
# export COMET_API_KEY=umDmt9f9wcCwCg1IqdPDXHX9G

#! visualizzazione su comet -> fare il login e tutto si vede in real time
comet_ml.init(project_name="yolov8s-FSOCO")
torch.cuda.set_device(0)

# yolov9c.pt oppure yolov9e.pt ---> (large/extra-large)
model = YOLO('yolov9c.pt')

# Controllo che il modello selezionato sia quello corretto
summary(model)


# batch=-1 regola dinamicamente la dimensione in base alla quantita di memoria disponbile nella GPU
# optimizer = 'auto'
results = model.train(
    data = '../data.yaml',
    imgsz = 640,
    epochs = 10,
    batch = 8,
    optimizer = 'auto',
    name = 'yolov9c_training_10_epoche',
    #project = 'comet-example-yolov8-FSOCO'
    project = '../result_detect_train'
)