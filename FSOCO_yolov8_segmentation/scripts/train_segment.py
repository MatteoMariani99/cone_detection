from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')

# batch=-1 regola dinamicamente la dimensione in base alla quantita di memoria disponbile nella GPU
# optimizer = 'auto'
results = model.train(
    data = '../data_seg.yaml',
    imgsz = 640,
    epochs = 10,
    batch = -1,
    optimizer = 'auto',
    name = 'yolov8n_seg_training_10_epoche',
    #project = 'comet-example-yolov8-FSOCO'
    project = '../result_segment_train/',
)
