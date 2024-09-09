# Cone detection tramite Yolov8
La repository contiene tutto il necessario per eseguire il training e testing per il riconoscimento dei coni gialli e blu tramite la rete Yolov8.

## Struttura
La repository è così strutturata:
- **FSOCO_yolov8_detect**: racchiude il training per l'identificazione dei coni con il dataset FSOCO e la rete yolov8;
- **FSOCO_yolov8_segmentation**: racchiude il training per la segmentazione dei coni con il dataset FSOCO e la rete yolov8;
- **FSOCO_yolov9_detect**: racchiude il training per l'identificazione dei coni con il dataset FSOCO e la rete yolov9;
- **install.sh**: bash script che permette la creazione di un ambiente conda per il training e il testing dei modelli yolo;

## Implementazione
La repository principale è **FSOCO_yolov8_detect** in quanto nel progetto si è deciso di effettuare l'identificazione dei coni tramite la rete yolov8: è stata inoltre eseguita una prova tramite la rete yolov9 non riscontrando però miglioramenti delle performance.
All'interno la repository presenta il file **data.yaml** c

