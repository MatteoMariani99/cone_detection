# Cone detection tramite Yolov8
La repository contiene tutto il necessario per eseguire il training e testing per il riconoscimento dei coni gialli e blu tramite la rete Yolov8.

## Descrizione
L'obiettivo è quello di allenare un modello della famiglia YOLO in modo da poterlo usare per l'identificazione coni all'interno dell'algoritmo Dagger. Il primo passo è stato quello di scegliere il dataset da utilizzare per l'allenamento: la scelta è ricaduta su [fsoco dataset](https://universe.roboflow.com/fmdv/fsoco-kxq3s), in quanto aveva già le immagini e le label predisposto per il training tramite rete yolo. Il modello scelto è **yolov8m.pt** ovvero il modello **medium** in modo da avere il giusto compromesso tra performance e dimensione della rete.


## Struttura 
La repository è così strutturata:
- **FSOCO_yolov8_detect**: racchiude il training per l'identificazione dei coni con il dataset FSOCO e la rete yolov8;
- **FSOCO_yolov8_segmentation**: racchiude il training per la segmentazione dei coni con il dataset FSOCO e la rete yolov8;
- **FSOCO_yolov9_detect**: racchiude il training per l'identificazione dei coni con il dataset FSOCO e la rete yolov9;
- **install.sh**: bash script che permette la creazione dell'ambiente conda per il training e il testing dei modelli yolo;

## Priorità delle cartelle

🟩 **FSOCO_yolov8_detect**: folder utilizzata in questo progetto;\
🟥 **FSOCO_yolov8_segmentation**: folder sperimentale, utilizzata solo per eseguire delle prove (NON USATA);\
🟥 **FSOCO_yolov9_detect**: folder sperimentale, utilizzata solo per confronto con la prima (NON USATA).

## Implementazione
Dopo aver eseguito lo script di installazione, oltre alla creazione dell'ambiente conda, verrà scaricato anche il dataset FSOCO: all'interno della cartella si troveranno tutte le immagini opportunamente splittate in train, test, valid e il file **data.yaml** che contiene tutte le informazioni necessarie da passare al modello per iniziare il training.\
La repository principale è **FSOCO_yolov8_detect** in quanto nel progetto si è deciso di effettuare l'identificazione dei coni tramite la rete yolov8: è stata inoltre eseguita una prova tramite la rete yolov9 non riscontrando però miglioramenti delle performance.\
Nella cartella **script** sono presenti gli script python per il training e il testing del modello: i risultati vengono poi salvati, in maniera ordinata per modello e numero di epoche, all'interno della cartella **result_detect_train**.\
Nella cartella **zed2** sono invece presenti gli script utili ad eseguire la cone detection direttamente con il modulo AI built-in della zed2 (è stato utilizzato come prova ma non fondamentale al fine del progetto).





## Installazione ✅
In primis è necessario eseguire il clone della repository tramite il comando:
```bash
git clone https://github.com/MatteoMariani99/cone_detection.git
```
❗**ATTENZIONE**❗: non è necessario eseguire il clone di questa repository se già è stato eseguito quella relativa all'algoritmo Dagger. Il clone va eseguito se si vuole utilizzare solamente questa repository!

Una volta eseguito il clone, aprire un terminale all'interno della repository tramite il comando `CTRL + ALT + T` ed eseguire:

```bash
chmod +x install.sh
```
in modo da fornire i privilegi di esecuzione dello script.

In seguito eseguire lo script tramite il comando:
```bash
bash install.sh
```
Inizierà così la creazione dell'ambiente conda con tutte le dipendenze necessarie per il funzionamento: verrà inoltre eseguito il download del dataset e salvato in un'apposita directory.
Una volta fatto ciò, l'ambiente deve essere attivato tramite il comando:
```bash
conda activate yolov8
```
