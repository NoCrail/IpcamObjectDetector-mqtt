import paho.mqtt.client as mqtt
import time
import random
import logging
import sys
import cv2
import numpy as np
from art import tprint
from threading import Thread
import threading
import requests
import configparser


# TO CONFIG
broker_url = "localhost"
broker_port = 1883
broker_user = "homeassistant"
broker_password = "pass"
mqtt_client_id = "watchdog"
topic_notifications = "custom/smartwatchdog/notifications_sw"
frame_rate = 2
# video = "yolo/videos/091309-av-1.mp4"
video = "rtsp://:@ipcam:554/stream0"
look_for = "person, bicycle, car, motorbike, train, truck, bird, dog, cat, backpack"
delay_after_recognition = 40
recognition_timeout = 600 
recognition_score = 0.5
recognition_times = 4
TOKEN = 'token'
URL = 'https://api.telegram.org/bot'
chat_id = "-4000000000"
FIRST_RECONNECT_DELAY = 1
RECONNECT_RATE = 2
MAX_RECONNECT_COUNT = 12
MAX_RECONNECT_DELAY = 60

notifications_enabled = False
topic_notifications_available = f"{topic_notifications}/available"
client_id = f'{mqtt_client_id}-{random.randint(0, 1000)}'
look_for = look_for.split(',')

logging.basicConfig(level=logging.DEBUG, filename="main.log",filemode="w",format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout)) # for logging to stdout 


def read_config():
    config = configparser.ConfigParser()
    config.read('config.conf')
    if config.has_option("MQTT", "broker_url"):
        broker_url = config["MQTT"]["broker_url"]
    if config.has_option("MQTT", "broker_port"):
        broker_port = config["MQTT"]["broker_port"]
    if config.has_option("MQTT", "broker_user"):
        broker_user = config["MQTT"]["broker_user"]
    if config.has_option("MQTT", "broker_password"):
        broker_password = config["MQTT"]["broker_password"]
    if config.has_option("MQTT", "mqtt_client_id"):
        mqtt_client_id = config["MQTT"]["mqtt_client_id"]
    if config.has_option("MQTT", "topic_notifications"):
        topic_notifications = config["MQTT"]["topic_notifications"]
    if config.has_option("video", "frame_rate"):
        frame_rate = config["video"]["frame_rate"]
    if config.has_option("video", "video"):
        logging.debug(f"Found video url {config['video']['video']}")
        video = config["video"]["video"]
    if config.has_option("video", "look_for"):
        look_for = config["video"]["look_for"]
    if config.has_option("video", "delay_after_recognition"):
        delay_after_recognition = config["video"]["delay_after_recognition"]
    if config.has_option("video", "alerts_timeout"):
        recognition_timeout = config["video"]["alerts_timeout"]
    if config.has_option("video", "recognition_score"):
        recognition_score = config["video"]["recognition_score"]
    if config.has_option("video", "recognition_times"):
        recognition_times = config["video"]["recognition_times"]
    if config.has_option("telegram", "TOKEN"):
        TOKEN = config["telegram"]["TOKEN"]
    if config.has_option("telegram", "chat_id"):
        chat_id = config["telegram"]["chat_id"]


# mqtt stuff
def mqtt_on_connect(client, userdata, flags, rc, properties):
    if rc == 0:
            logging.info("Connected to MQTT Broker!")
            set_available(client, True)
    else:
            logging.error("Failed to connect, return code %d\n", rc)
def mqtt_client_stop(client):
    set_available(client, False)
    logging.info("Stopping mqtt client...")
    client.loop_stop() 
    client.disconnect()
def mqtt_on_disconnect(client, userdata, flags, rc, properties):
    logging.warning("Disconnected with result code: %s", rc)
    if rc != 0:
        reconnect_count, reconnect_delay = 0, FIRST_RECONNECT_DELAY
        while reconnect_count < MAX_RECONNECT_COUNT:
            logging.info("Reconnecting in %d seconds...", reconnect_delay)
            time.sleep(reconnect_delay)

            try:
                client.reconnect()
                logging.info("Reconnected successfully!")
                mqtt_client_init(client)
                return
            except Exception as err:
                logging.error("%s. Reconnect failed. Retrying...", err)

            reconnect_delay *= RECONNECT_RATE
            reconnect_delay = min(reconnect_delay, MAX_RECONNECT_DELAY)
            reconnect_count += 1
        logging.error("Reconnect failed after %s attempts. Exiting...", reconnect_count)
def mqtt_client_init(client=None):
    if not client:
        logging.info("No client found, creating...")
        client = mqtt.Client(client_id=client_id, callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    logging.debug(f"Mqtt client_id:{client_id}")
    client.username_pw_set(broker_user, broker_password)
    client.on_connect = mqtt_on_connect
    client.on_disconnect = mqtt_on_disconnect
    client.connect(broker_url, broker_port)
    mqtt_subscribe(client)
    return client
def mqtt_on_message(client, userdata, msg):
        global notifications_enabled
        logging.debug(f"Received `{msg.payload.decode()}` from `{msg.topic}` topic")
        if msg.payload.decode() == 'ON':
            notifications_enabled = True
            logging.info(f"Notifications enabled after mqtt message")
        elif msg.payload.decode() == 'OFF':
            notifications_enabled = False
            logging.info(f"Notifications disabled after mqtt message")
def mqtt_subscribe(client):
    logging.debug(f"Subscribing to topic {topic_notifications}")
    client.subscribe(topic_notifications)
    client.on_message = mqtt_on_message


# service stuff
def set_available(client, state):
    logging.debug(f"Setting available to {state}")
    if state:
        client.publish(topic_notifications_available, payload="online", qos=0, retain=False)
    else:
        client.publish(topic_notifications_available, payload="offline", qos=0, retain=False)
def set_enabled(client):
    logging.debug(f"Setting enabled to {notifications_enabled}")
    if notifications_enabled:
        client.publish(topic_notifications, payload="ON", qos=0, retain=False)
    else:
        client.publish(topic_notifications, payload="OFF", qos=0, retain=False)


#video stuff
def apply_yolo_object_detection(image_to_process):

    """
    Recognition and determination of the coordinates of objects on the image
    :param image_to_process: original image
    :return: image with marked objects and captions to them
    """

    height, width, _ = image_to_process.shape
    blob = cv2.dnn.blobFromImage(image_to_process, 1 / 255, (608, 608),
                                 (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(out_layers)
    class_indexes, class_scores, boxes = ([] for i in range(3))
    objects_count = 0
    recognition_result = {}
    # Starting a search for objects in an image
    for out in outs:
        for obj in out:
            scores = obj[5:]
            class_index = np.argmax(scores)
            class_score = scores[class_index]
            if class_score > recognition_score:
                recognition_result[classes[class_index]]=class_score
                # print(f"{class_index} score:{class_score}")
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                obj_width = int(obj[2] * width)
                obj_height = int(obj[3] * height)
                box = [center_x - obj_width // 2, center_y - obj_height // 2,
                       obj_width, obj_height]
                boxes.append(box)
                class_indexes.append(class_index)
                class_scores.append(float(class_score))

    # Selection
    chosen_boxes = cv2.dnn.NMSBoxes(boxes, class_scores, 0.0, 0.4)
    for box_index in chosen_boxes:
        box_index = box_index
        box = boxes[box_index]
        class_index = class_indexes[box_index]

        # For debugging, we draw objects included in the desired classes
        if classes[class_index] in classes_to_look_for:
            objects_count += 1
            image_to_process = draw_object_bounding_box(image_to_process,
                                                        class_index, box)

    final_image = image_to_process
    # final_image = draw_object_count(image_to_process, objects_count)
    # print(recognition_result)
    return recognition_result, final_image
def draw_object_bounding_box(image_to_process, index, box):
    """
    Drawing object borders with captions
    :param image_to_process: original image
    :param index: index of object class defined with YOLO
    :param box: coordinates of the area around the object
    :return: image with marked objects
    """

    x, y, w, h = box
    start = (x, y)
    end = (x + w, y + h)
    color = (0, 255, 0)
    width = 2
    final_image = cv2.rectangle(image_to_process, start, end, color, width)

    start = (x, y - 10)
    font_size = 1
    font = cv2.FONT_HERSHEY_SIMPLEX
    width = 2
    text = classes[index]
    final_image = cv2.putText(final_image, text, start, font,
                              font_size, color, width, cv2.LINE_AA)

    return final_image
def draw_object_count(image_to_process, objects_count):
    """
    Signature of the number of found objects in the image
    :param image_to_process: original image
    :param objects_count: the number of objects of the desired class
    :return: image with labeled number of found objects
    """

    start = (10, 120)
    font_size = 1.5
    font = cv2.FONT_HERSHEY_SIMPLEX
    width = 3
    text = "Objects found: " + str(objects_count)

    # Text output with a stroke
    # (so that it can be seen in different lighting conditions of the picture)
    white_color = (255, 255, 255)
    black_outline_color = (0, 0, 0)
    final_image = cv2.putText(image_to_process, text, start, font, font_size,
                              black_outline_color, width * 3, cv2.LINE_AA)
    final_image = cv2.putText(final_image, text, start, font, font_size,
                              white_color, width, cv2.LINE_AA)

    return final_image

def start_video_object_detection(video: str):
    """
    Real-time video capture and analysis
    """

    while True:
        try:
            # Capturing a picture from a video
            prev = 0
            video_camera_capture = cv2.VideoCapture(video)
            
            while video_camera_capture.isOpened():
                time_elapsed = time.time() - prev
                ret, frame = video_camera_capture.read()
                if time_elapsed > 1./frame_rate:
                    prev = time.time()
                    
                    if not ret:
                        break
                    
                    # Application of object recognition methods on a video frame from YOLO
                    frame = apply_yolo_object_detection(frame)
                    

            
            video_camera_capture.release()
            cv2.destroyAllWindows()
    
        except KeyboardInterrupt:
            break
            # pass

def sendImage(rec):
    video_camera_capture = cv2.VideoCapture(video)
    ret, frame = video_camera_capture.read()
    if video_camera_capture.isOpened():
        ret,frame = video_camera_capture.read()
        video_camera_capture.release()
    
    if ret and frame is not None:
        img = 'images/latest.jpg'
        cv2.imwrite(img, frame)
        files={}
        files[f"photo"] =  open(img, 'rb')
        data = {"caption": f'Обнаружены объекты: {rec}'}
        requests.post(f'{URL}{TOKEN}/sendPhoto?chat_id={chat_id}', files=files, data=data)
    logging.info("Sending pic")


class VideoScreenshot(object):
    def __init__(self, src=0):
        # Create a VideoCapture object
        self.capture = cv2.VideoCapture(src)

        # Take screenshot every x seconds
        self.screenshot_interval = 1

        # Default resolutions of the frame are obtained (system dependent)
        self.frame_width = int(self.capture.get(3))
        self.frame_height = int(self.capture.get(4))

        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()

    def show_frame(self):
        # Display frames in main program
        if self.status:
            # print(self.frame)
            cv2.imshow('frame', self.frame)

        # Press Q on keyboard to stop recording
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.capture.release()
            cv2.destroyAllWindows()
            exit(1)

    def rec_frame(self):
        # Display frames in main program
        if self.status:
            # cv2.imshow('frame', self.frame)
            res_rec, res_fr = apply_yolo_object_detection(self.frame)
            
            # print(res_rec)
            return res_rec

        # Press Q on keyboard to stop recording
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.capture.release()
            cv2.destroyAllWindows()
            exit(1)

    def save_frame(self): #bugging, creates many pics
        # Save obtained frame periodically
        self.frame_count = 1
        def save_frame_thread():
            while True:
                try:
                    cv2.imwrite('frame_{}.png'.format(self.frame_count), self.frame)
                    self.frame_count += 1
                    time.sleep(self.screenshot_interval)
                except AttributeError:
                    pass
        Thread(target=save_frame_thread, args=()).start()



#read_config()
net = cv2.dnn.readNetFromDarknet("yolov/yolov3.cfg",
                                     "yolov/yolov3.weights")
layer_names = net.getLayerNames()
out_layers_indexes = net.getUnconnectedOutLayers()
out_layers = [layer_names[index - 1] for index in out_layers_indexes]

# Loading from a file of object classes that YOLO can detect
with open("yolov/yolov3.txt") as file:
    classes = file.read().split("\n")
# Delete spaces
list_look_for = []
for look in look_for:
    list_look_for.append(look.strip())
classes_to_look_for = list_look_for

notifications_enabled = True
client = mqtt_client_init()
set_enabled(client)


video_stream_widget = VideoScreenshot(video)
# video_stream_widget.save_frame()

last_recognition = 0
recognised = 0

send_thread = threading.Thread(target=sendImage, name="Sender")
logging.debug(f"look_for: {classes_to_look_for}")
while True:
        
        try:
            client.loop(timeout=0.2)
            # video_stream_widget.show_frame()
            rec = video_stream_widget.rec_frame()
            # Capturing a picture from a video
            
            if bool(rec): # recognition list not empty
                found = False
                for it in rec.keys():
                    if it in classes_to_look_for:
                        found = True
                if found:
                    recognised = recognised + 1
                    logging.debug(f"Got not empty recognition list, rocgnised={recognised}, list={rec}")
                    if time.time() - last_recognition > recognition_timeout:
                        logging.debug(f"No recognition timeout, time:{time.time()}, last:{last_recognition}, timeout:{recognition_timeout}")
                        if recognised >= recognition_times: #recognised multiple times
                            logging.debug(f"Recognised enough times {recognised}")
                            last_recognition = time.time()
                            if notifications_enabled: # enabled notifications
                                # TODO delay or maybe play with recognition times
                                logging.info(f"Recognition success! Found {rec}")
                                # sendImage(video)
                                send_thread = threading.Thread(target=sendImage, name="Sender", args=(rec))
                                send_thread.start()
                            else:
                              logging.info(f"Notifications disabled")
                    else:
                        logging.debug(f"Recognition timeout time:{time.time()}, last:{last_recognition}, timeout:{recognition_timeout}")

            else:
                logging.debug(f"Got empty list, resetting recognition times")
                recognised = 0

            # print(rec)

        except AttributeError:
            pass

        except KeyboardInterrupt:
            break
            # pass


notifications_enabled = False
set_enabled(client)
mqtt_client_stop(client)
