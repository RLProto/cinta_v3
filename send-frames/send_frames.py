import cv2
import numpy as np
import time
import asyncio
import aiohttp
import os
import logging
import threading
import queue

# Constants for the APIs and Node-RED
URL_8888 = os.getenv('URL_8888', 'http://localhost:8888')
URL_9999 = os.getenv('URL_9999', 'http://localhost:9999')
NODE_RED_URL = os.getenv('NODE_RED_URL', 'http://localhost:1880/cinta')
INFERENCE_ENDPOINT = "/inference"
UPLOAD_MODEL_ENDPOINT = "/upload-model"
INFERENCE_INTERVAL = int(os.getenv('INFERENCE_INTERVAL', 15))
CAMERA_URL = os.getenv('CAMERA_URL')

# Ensure the directory for saving frames exists
save_dir = "/app/images"
os.makedirs(save_dir, exist_ok=True)

# Define a custom logging level
IMPORTANT = 25
logging.addLevelName(IMPORTANT, "IMPORTANT")

def important(self, message, *args, **kws):
    if self.isEnabledFor(IMPORTANT):
        self._log(IMPORTANT, message, args, **kws)

# Add the custom level to the Logger class
logging.Logger.important = important

# Set up logging to use the custom level
logging.basicConfig(level=IMPORTANT, format='%(asctime)s - %(levelname)s - %(message)s')

class VideoCapture:
    def __init__(self, url):
        self.url = url
        self.connect()
        self.q = queue.Queue(maxsize=10)
        t = threading.Thread(target=self._reader)
        t.daemon = True
        t.start()

    def connect(self):
        self.cap = cv2.VideoCapture(self.url)
        if not self.cap.isOpened():
            logging.error("Failed to connect to the camera.")

    def reconnect(self):
        self.cap.release()
        self.connect()
        if self.cap.isOpened():
            logging.getLogger().important("Successfully reconnected to the camera.")
        else:
            logging.error("Failed to reconnect to the camera.")

    def release(self):
        if self.cap is not None:
            self.cap.release()

    def _reader(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                logging.critical("Camera disconnected. Attempting to reconnect.")
                self.reconnect()
                time.sleep(5)  # Wait for 5 seconds before attempting to read again
                continue  # Skips the rest of the loop and jumps to the next iteration
            if not self.q.empty():
                try:
                    self.q.get_nowait()   # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put((ret, frame))

    def read(self):
        return self.q.get()

async def send_request(session, url, img_encoded):
    headers = {'Content-Type': 'application/octet-stream'}
    async with session.post(url, data=img_encoded.tobytes(), headers=headers) as response:
        if response.status == 200:
            return await response.json()
        else:
            logging.error(f"Failed to get a valid response from {url}: {response.status}")
            return None

async def send_to_node_red(session, data):
    try:
        async with session.post(NODE_RED_URL, json=data, timeout=3) as response:
            if response.status != 200:
                logging.error(f"Failed to send data to Node-RED: {response.status}, {await response.text()}")
    except Exception as e:
        logging.error(f"Failed to send data to Node-RED: {str(e)}")

def annotate_frame(frame, text, position, rect_color, text_color):
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    cv2.rectangle(frame, (position[0], position[1] - text_height - 10), 
                  (position[0] + text_width, position[1] + 10), rect_color, -1)
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA)

async def upload_model(session, url, file_path):
    with open(file_path, 'rb') as file:
        files = {'file': file}
        response = await session.post(url, files=files)
        if response.status != 200:
            logging.error(f"Failed to upload model to {url}: {response.status}")
        else:
            logging.info(f"Model uploaded successfully to {url}")

async def main():
    cap = VideoCapture(CAMERA_URL)
    async with aiohttp.ClientSession() as session:
        # Upload models
        await upload_model(session, URL_9999 + UPLOAD_MODEL_ENDPOINT, 'model/ag.zip')
        await upload_model(session, URL_8888 + UPLOAD_MODEL_ENDPOINT, 'model/cinta_.zip')

        while True:
            ret, frame = cap.read()
            if not ret:
                logging.error("Failed to read frame from camera.")
                continue

            # Processing regions of interest
            x1_graos, y1_graos, x2_graos, y2_graos = 675, 345, 1160, 1080
            x1_ag, y1_ag, x2_ag, y2_ag = 745, 280, 955, 480
            cropped_frame_graos = frame[y1_graos:y2_graos, x1_graos:x2_graos]
            cropped_frame_ag = frame[y1_ag:y2_ag, x1_ag:x2_ag]

            # Draw rectangle around ROI for graos
            cv2.rectangle(frame, (x1_graos, y1_graos), (x2_graos, y2_graos), (0, 255, 0), 3)

            # Encoding and sending the frame for classification
            _, img_encoded = cv2.imencode('.jpg', cropped_frame_graos)
            response = await send_request(session, URL_8888 + INFERENCE_ENDPOINT, img_encoded)
            if response:
                cinta_class = response.get('classification')
                cinta_conf = response.get('confidence-score')

                # Annotate and save frame for cinta classification
                text_cinta = f"cinta: {cinta_class}, Confidence: {cinta_conf}%"
                annotate_frame(frame, text_cinta, (50, 40), (255, 255, 255), (0, 0, 255))
                cv2.imwrite(os.path.join(save_dir, f"cinta_{int(time.time())}.jpg"), frame)

                if cinta_class == "alto" and cinta_conf >= 98:
                    await asyncio.sleep(1)  # Delay for capturing a new frame
                    ret, frame = cap.read()
                    if not ret:
                        logging.error("Failed to read frame after delay.")
                        continue

                    # Draw rectangle around ROI for ag
                    cv2.rectangle(frame, (x1_ag, y1_ag), (x2_ag, y2_ag), (0, 255, 0), 3)

                    _, img_encoded = cv2.imencode('.jpg', cropped_frame_ag)
                    response_ag = await send_request(session, URL_9999 + INFERENCE_ENDPOINT, img_encoded)
                    if response_ag:
                        ag_class = response_ag.get('classification')
                        ag_conf = response_ag.get('confidence-score')

                        # Annotate and save frame for AG classification
                        text_ag = f"AG: {ag_class}, Confidence: {ag_conf}%"
                        annotate_frame(frame, text_ag, (50, 90), (255, 255, 255), (255, 0, 0))
                        cv2.imwrite(os.path.join(save_dir, f"ag_{int(time.time())}.jpg"), frame)

                        # Data package for Node-RED from the second classifier
                        node_red_data_ag = {
                            "cinta_classification": cinta_class,
                            "cinta_confidence_score": cinta_conf,
                            "ag_classification": ag_class,
                            "ag_confidence_score": ag_conf
                        }
                        logging.getLogger().important(f"Data: {node_red_data_ag}")
                        asyncio.create_task(send_to_node_red(session, node_red_data_ag))
                else:
                    # Data package for Node-RED from the first classifier
                    node_red_data_cinta = {
                    "cinta_classification": cinta_class,
                    "cinta_confidence_score": cinta_conf
                    }
                    logging.getLogger().important(f"Data: {node_red_data_cinta}")
                    asyncio.create_task(send_to_node_red(session, node_red_data_cinta))
            await asyncio.sleep(INFERENCE_INTERVAL)  # Wait for the next interval

if __name__ == "__main__":
    asyncio.run(main())
