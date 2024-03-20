import comet_ml
from ultralytics import YOLO
from torchinfo import summary
import torch


# nel terminale
# export COMET_API_KEY=umDmt9f9wcCwCg1IqdPDXHX9G

#! visualizzazione su comet -> fare il login e tutto si vede in real time
#comet_ml.init(project_name="comet-example-yolov8-FSOCO")
torch.cuda.set_device(0)

yoloModel = "yolov8n.pt"

# # selezionare il modello che si desidera allenare
# user_input = int(input("Selezionare il modello da allenare: \n1: yolo nano\n2: yolo small\n3: yolo medium\n4: yolo large\n5: yolo extra-large\nScelta: "))

# if user_input==1:
#     model = YOLO('yolov8n.pt')
# elif user_input==2:
#     model = YOLO('yolov8s.pt')
# elif user_input==3:
#     model = YOLO('yolov8m.pt')
# elif user_input==4:
#     model = YOLO('yolov8l.pt')
# elif user_input==5:
#     model = YOLO('yolov8x.pt')
# else:
#     print("Modello non disponibile!")

# print("Modello selezionato: ",user_input)


model = YOLO(yoloModel,'gpu')

# Controllo che il modello selezionato sia quello corretto
summary(model)

# batch=-1 regola dinamicamente la dimensione in base alla quantita di memoria disponbile nella GPU
# optimizer = 'auto'
results = model.train(
    data = '../data.yaml',
    imgsz = 640,
    epochs = 10,
    batch = -1,
    optimizer = 'auto',
    name = 'prova',
    #project = 'comet-example-yolov8-FSOCO'
    project = '../result_detect_train'
)