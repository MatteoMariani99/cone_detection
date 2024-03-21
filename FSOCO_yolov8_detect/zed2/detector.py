#!/usr/bin/env python3

import numpy as np

import argparse
import torch
import cv2
import pyzed.sl as sl # API python per la zed


from ultralytics import YOLO

from threading import Lock, Thread
from time import sleep

import ogl_viewer.viewer as gl
import cv_viewer.tracking_viewer as cv_viewer

lock = Lock()
run_signal = False
exit_signal = False
inference_time = []



def xywh2abcd(xywh):
    output = np.zeros((4, 2))

    # Center / Width / Height -> BBox corners coordinates
    # xywh sono:
    # xy coordinate del centro del bounding box
    # wh larghezza e altezza
    # Così facendo sto trovando i punti medi dei lati del rettangolo (asterischi)
    # A --*-- B
    # |        |
    # * Obj    *
    # |        |
    # D --*-- C

    x_min = (xywh[0] - 0.5*xywh[2]) #* im_shape[1]
    x_max = (xywh[0] + 0.5*xywh[2]) #* im_shape[1]
    y_min = (xywh[1] - 0.5*xywh[3]) #* im_shape[0]
    y_max = (xywh[1] + 0.5*xywh[3]) #* im_shape[0]

    # per trovare i vertici basta raggruppare le coordinate
    # A ------ B
    # | Object |
    # D ------ C

    output[0][0] = x_min
    output[0][1] = y_min

    output[1][0] = x_max
    output[1][1] = y_min

    output[2][0] = x_max
    output[2][1] = y_max

    output[3][0] = x_min
    output[3][1] = y_max
    
    # output mi ritorna le coordinate dei 4 vertici
    return output


def detections_to_custom_box(detections):
    output = []
    # enumerate conteggia le tuple presenti in detections
    for i, det in enumerate(detections):
        # estrapolo i valori delle coordinate del centro box + larghezza e altezza
        xywh = det.xywh[0]
        
        # Creazione oggetto
        # Creating ingestable objects for the ZED SDK
        # https://www.stereolabs.com/docs/api/python/classpyzed_1_1sl_1_1CustomBoxObjectData.html
        obj = sl.CustomBoxObjectData()
        obj.bounding_box_2d = xywh2abcd(xywh) # sono quelli calcolati con la funzione sopra
        obj.label = det.cls
        obj.probability = det.conf
        obj.is_grounded = False # l'oggetto non si muove sul pavimento
        output.append(obj)
    return output


def torch_thread(img_size, conf_thres=0.8, iou_thres=0.47):
    global image_net, exit_signal, run_signal, detections

    print("Intializing Network...")
    
    # setting gpu
    torch.cuda.set_device(0)


    # va copiato il modello migliore nel folder per poterlo riusare
    model = YOLO('best.pt','gpu')

   

    while not exit_signal:
        if run_signal:
            lock.acquire()

            # tolgo il canale a in modo da passare al modello un'immagine con 3 canali
            img = cv2.cvtColor(image_net, cv2.COLOR_RGBA2RGB)
            
            
            #* PREDICT
            # https://docs.ultralytics.com/modes/predict/#video-suffixes
            result = model.predict(img, save=False, imgsz=img_size, conf=conf_thres, iou=iou_thres)
            speed = result[0].speed
            inference_time.append(speed['inference'])
            print("Tempo medio di inferenza: ",sum(inference_time)/len(inference_time))
            # for box in result[0].boxes:
            #     class_id = int(box.cls)  # Get class ID
            #     class_label = result[0].names[class_id]  # Get class label from class ID
            #     class_label_list.insert(0,class_label)
            
            det = result[0].boxes
            
            # ritorna l'oggetto contenente le informazioni relative alla detection (bouinding box, classi, conf)
            detections = detections_to_custom_box(det)
            lock.release()
            run_signal = False
        sleep(0.01)


def main():
    global image_net, exit_signal, run_signal, detections
    

    capture_thread = Thread(target=torch_thread, kwargs={'img_size': opt.img_size, "conf_thres": opt.conf_thres})
    capture_thread.start()

    print("Initializing Camera...")

    # creazione oggetto CAMERA
    zed = sl.Camera()

    # riguardo l'argparse
    input_type = sl.InputType()
    if opt.svo is not None:
        input_type.set_from_svo_file(opt.svo)

    #* init parametri
    # Creazione dell'oggetto dei parametri di inizializzazione e setting della configurazione dei parametri
    # https://www.stereolabs.com/docs/api/python/classpyzed_1_1sl_1_1DEPTH__MODE.html
    # https://www.stereolabs.com/docs/api/python/classpyzed_1_1sl_1_1OBJECT__DETECTION__MODEL.html
    # https://www.stereolabs.com/docs/api/python/classpyzed_1_1sl_1_1UNIT.html
    # https://www.stereolabs.com/docs/api/python/classpyzed_1_1sl_1_1COORDINATE__SYSTEM.html
    init_params = sl.InitParameters(input_t=input_type, svo_real_time_mode=True)
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # QUALITY
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.depth_maximum_distance = 50

    runtime_params = sl.RuntimeParameters()
    
    #* apro il flusso video
    status = zed.open(init_params)

    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    image_left_tmp = sl.Mat() # crea una matrice

    print("Initialized Camera")

    #* Inizializzazione del modulo che permette di fare il tracking del movimento degli oggetti nell'ambiente
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    # If the camera is static, uncomment the following line to have better performances and boxes sticked to the ground.
    # positional_tracking_parameters.set_as_static = True
    zed.enable_positional_tracking(positional_tracking_parameters)
    

    #* Modulo di object detection
    # Viene definito il modulo che conterrà tutti i parametri relativi al object detection
    obj_param = sl.ObjectDetectionParameters()
    obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS # utilizzo di un modello custom
    obj_param.enable_tracking = True
    # attivazione modulo di object detection
    zed.enable_object_detection(obj_param)

    # classe che contiene i risultati dell'object detection
    objects = sl.Objects()
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()


    #* Display
    # Struttura contenente informazioni sulla camera (numero seriale, modello, calibrazion...)
    camera_infos = zed.get_camera_information()
    camera_res = camera_infos.camera_configuration.resolution # ottengo informazioni sulla risoluzione
  
    
    #* Apertura OpenGL viewer, definizione point cloud e TrackingViewer
    viewer = gl.GLViewer()
    point_cloud_res = sl.Resolution(min(camera_res.width, 720), min(camera_res.height, 404))
    point_cloud_render = sl.Mat()
    viewer.init(camera_infos.camera_model, point_cloud_res, obj_param.enable_tracking)
    # sl.MAT_TYPE.F32_C4: matrice di float a 4 canali
    point_cloud = sl.Mat(point_cloud_res.width, point_cloud_res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
    
    # si acquisisce l'immagine proveniente dalla lente di sinistra
    image_left = sl.Mat()
    display_resolution = sl.Resolution(min(camera_res.width, 1280), min(camera_res.height, 720))
    image_scale = [display_resolution.width / camera_res.width, display_resolution.height / camera_res.height]
    image_left_ocv = np.full((display_resolution.height, display_resolution.width, 4), [245, 239, 239, 255], np.uint8)

    # Info per la visualizzazione del tracking
    camera_config = camera_infos.camera_configuration
    tracks_resolution = sl.Resolution(400, display_resolution.height)
    track_view_generator = cv_viewer.TrackingViewer(tracks_resolution, camera_config.fps, init_params.depth_maximum_distance)
    track_view_generator.set_camera_calibration(camera_config.calibration_parameters)
    image_track_ocv = np.zeros((tracks_resolution.height, tracks_resolution.width, 4), np.uint8)
    
    #* Camera pose: contiene dati di posizione per il tracking data la posizione e orientazione 3d nello spazio della camera
    cam_w_pose = sl.Pose()


    #* Check per verificare se è possibile aprire il viewer.
    #* Per catturare e processare un'immagine è necessario chiamare la funzione grab()
    while viewer.is_available() and not exit_signal:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # -- Get the image
            lock.acquire()
            # acquisisco l'immagine dalla lente di sinistra e la salvo nella matrice image_left_tmp
            zed.retrieve_image(image_left_tmp, sl.VIEW.LEFT)
            image_net = image_left_tmp.get_data() # ritorna un numpy array contenente l'immagine
            lock.release()
            run_signal = True

            # -- Detection running on the other thread
            while run_signal:
                sleep(0.001)

            # Wait for detections
            lock.acquire()
            # -- Ingest detections
            zed.ingest_custom_box_objects(detections) # si occupa di mettere i box
            lock.release()
            # salva tutti gli oggetti riconosciuti nel "oggetto python" objects
            zed.retrieve_objects(objects, obj_runtime_param)
            
            
            #* Display deep
            # Se si vuole ottenere la posizione di un punto in metri nell'immagine, basta calcolare le misure di depth
            # con la funzione retrieve_measure e poi estrapolarne il valore con get_value
            #! scommentare le seguenti righe per farlo
            # depth_map = sl.Mat()
            # zed.retrieve_measure(depth_map, sl.MEASURE.DEPTH) # Get the depth map
    
            # # Read a depth value
            #x = int(camera_res.width / 2) # Center coordinates
            #y = int(camera_res.height / 2)
 
            # err, center_depth = depth_map.get_value(x, y) # each depth map pixel is a float value
            # if err == sl.ERROR_CODE.SUCCESS: # + Inf is "too far", -Inf is "too close", Nan is "unknown/occlusion"
            #     print("Depth value at center:", center_depth, init_params.coordinate_units)
                
                
            #* Display point cloud
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, point_cloud_res)
            point_cloud.copy_to(point_cloud_render)
            # Se si vuole leggere un valore specifico, utilizzare la funzione get_value e stampare le 3 coordinate del valore
            #! scommentare le seguenti righe per farlo
            # err, pc_value = point_cloud.get_value(x, y) # each point cloud pixel contains 4 floats, so we are using a numpy array
            
            # # Get 3D coordinates
            # if err == sl.ERROR_CODE.SUCCESS:
            #     print("Point cloud coordinates at center: X=", pc_value[0], ", Y=", pc_value[1], ", Z=", pc_value[2])
        
            # si ottiene l'immagine e la si salva in image_left
            zed.retrieve_image(image_left, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
            # fornisce la posizione della camera rispetto al frame world: quando la camera si muove, verrà restituita la differenza di posizione
            # rispetto al suo punto di partenza.
            # si può usare anche FRAME.CAMERA che fornisce la differenza di posizione tra dove si trova la camera ora e la sua posizione precedente
            zed.get_position(cam_w_pose, sl.REFERENCE_FRAME.WORLD)
            translation = cam_w_pose.get_translation().get()
            print("Camera position: X=", translation[0], " Y=", translation[1], " Z=", translation[2])

            # 3D rendering
            viewer.updateData(point_cloud_render, objects)
            # 2D rendering
            np.copyto(image_left_ocv, image_left.get_data())
            
            
            
            cv_viewer.render_2D(image_left_ocv, image_scale, objects,obj_param.enable_tracking)
            global_image = cv2.hconcat([image_left_ocv, image_track_ocv])
            # Tracking view
            track_view_generator.generate_view(objects, cam_w_pose, image_track_ocv, objects.is_tracked)

            cv2.imshow("ZED | 2D View and Birds View", global_image)
            key = cv2.waitKey(10)
            if key == 27:
                exit_signal = True
        else:
            exit_signal = True

    viewer.exit()
    exit_signal = True
    zed.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov8m.pt', help='model.pt path(s)')
    parser.add_argument('--svo', type=str, default=None, help='optional svo file')
    parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.4, help='object confidence threshold')
    opt = parser.parse_args()

    with torch.no_grad():
        main()
