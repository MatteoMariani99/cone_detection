from ultralytics import YOLO
import torch
import cv2

# setting gpu
device = torch.cuda.set_device(0)


# va copiato il modello migliore nel folder per poterlo riusare
model = YOLO('best.pt', 'gpu')

inference_time = []

cap = cv2.VideoCapture(2)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        #results = model.track(frame, persist=True)
        results = model.predict(frame)
        speed = results[0].speed
        inference_time.append(speed['inference'])
        print("Tempo medio di inferenza: ",sum(inference_time)/len(inference_time))
        
        # dentro in result [0] ci sono tante info tra cui i boxes
        # print(results[0].boxes)

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


# for result in results:
#     boxes = result.boxes
#     masks = result.masks
#     keypoints = result.keypoints
#     probs = result.probs
#     result.show()
#     result.save(filename = 'result1.jpg')