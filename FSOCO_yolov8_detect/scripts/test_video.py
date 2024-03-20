#import supervision as sv
import numpy as np
from ultralytics import YOLO
import cv2
import torch

# setting gpu
torch.cuda.set_device(0)

model = YOLO('best.pt','gpu') # modello migliore durante il training


# video direttamente da youtube
# botsort o bytetrack -> da migliori risultati il primo
# classes = [0,1,2,3,4]
#results = model.track(source='https://youtu.be/r8QIRiEfIS0',save=False, show=True, tracker="botsort.yaml",classes=classes)



#! OPPURE #


# Load the YOLOv8 model
#    model = YOLO('yolov8n.pt')

# per scaricare un video da youtube da terminale
# yt-dlp https://youtu.be/r8QIRiEfIS0 myvideo.mp4
#video_path = "https://youtu.be/r8QIRiEfIS0"
#    video = pafy.new(video_path)
#    best  = video.getbest(preftype="webm")
#documentation: https://pypi.org/project/pafy/

#cap = cv2.VideoCapture('myvideo.mp4')
cap = cv2.VideoCapture(0)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        #results = model.track(frame, persist=True)
        results = model.predict(frame)
        
        # dentro in result [0] ci sono tante info tra cui i boxes
        # print(results[0].boxes)
        # posso ora fare la differenza con i boxes che trovo nella cartella labels del train e vedere quanto sono diversi

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()


# Prova segmentazione: YOLOv8n-Seg modello

# Box in termini di vertice in termini di:
# [verticebassosx_x, verticebassosx_y, larghezza rettangolo, altezza rettangolo]
# xywh: tensor([[ 189.1328,  524.6973,   69.0652,   80.9896],
#     [ 762.7911,  417.6235,   23.2559,   27.7196],
#     [ 433.6178,  420.0963,   21.9335,   26.7374],
#     [1115.2092,  516.2863,   48.2383,   76.3828]]]

# Box in termini di vertice in termini di:
# [verticealtosx_x, verticealtosx_y, verticeabassodx_x, verticeabassodx_y]
# xyxy: tensor([[ 154.6002,  484.2025,  223.6654,  565.1921],
# [ 751.1632,  403.7637,  774.4191,  431.4833],
# [ 422.6511,  406.7276,  444.5845,  433.4649],
# [1091.0901,  478.0949, 1139.3284,  554.4777]]]