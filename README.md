# Cone detection tramite Yolov8
La repository contiene tutto il necessario per eseguire il training e testing per il riconoscimento dei coni gialli e blu tramite la rete Yolov8.

## Struttura
La repository è così strutturata:
- **FSOCO_yolov8_detect**: racchiude il training per l'identificazione dei coni con il dataset FSOCO e la rete yolov8;
- **FSOCO_yolov8_segmentation**: racchiude il training per la segmentazione dei coni con il dataset FSOCO e la rete yolov8;
- **FSOCO_yolov9_detect**: racchiude il training per l'identificazione dei coni con il dataset FSOCO e la rete yolov9;
- **install.sh**: bash script che permette la creazione di un ambiente conda per il training e il testing dei modelli yolo;
- **dataset_setting.png**: descrive come specificare il dataset utilizzato per il training della rete.

## Implementazione
La repository principale è **FSOCO_yolov8_detect** in quanto nel progetto si è deciso di effettuare l'identificazione dei coni tramite la rete yolov8: è stata inoltre eseguita una prova tramite la rete yolov9 non riscontrando però miglioramenti delle performance.
All'interno la repository presenta il file **data.yaml** che contiene tutte le informazioni necessarie da passare al modello per iniziare il training.
Nella cartella **script** sono presenti gli script python per il training e il testing del modello: i risultati vengono poi salvati, in maniera ordinata per modello e numero di epoche, all'interno della cartella **result_detect_train**.
Nella cartella **zed2** sono invece presenti gli script utili ad eseguire la cone detection direttamente con il modulo AI built-in della zed2 (è stato utilizzato come prova ma non fondamentale al fine del progetto).

