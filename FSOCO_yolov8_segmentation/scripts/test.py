from ultralytics import YOLO
import torch

# setting gpu
device = torch.cuda.set_device(0)


# va copiato il modello migliore nel folder per poterlo riusare
model = YOLO('best.pt', 'gpu')


# testo un'immagine
# per avere il real time del video (sorgente 2 era la zed2)
results = model.predict(source=0, save=False, show = True, conf=0.43)


# for result in results:
#     boxes = result.boxes
#     masks = result.masks
#     keypoints = result.keypoints
#     probs = result.probs
#     result.show()
#     result.save(filename = 'result1.jpg')