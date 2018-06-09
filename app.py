from PyQt5 import QtCore, QtWidgets, uic, QtGui
import sys
import cv2
import threading
from queue import *
from PIL import Image
import snack_detection
import employee_detection
import image_processing

running = False
capture_thread = None
form_class = uic.loadUiType("ui/simple.ui")[0]
q = Queue()

#main app

def start_all_threads():
    capture_thread.start()
    ocr_thread.start()
    snack_detection_thread.start()

def stop_all_threads():
    capture_thread.stop()
    ocr_thread.stop()
    snack_detection_thread.stop()

def start_thread(thread):
    thread.start()

def stop_thread(thread):
    thread.stop()

# QT widget for showing image
class OwnImageWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(OwnImageWidget, self).__init__(parent)
        self.image = None

    def setImage(self, image):
        self.image = image
        sz = image.size()
        self.setMinimumSize(sz)
        self.update()

    def paintEvent(self, event):
        qp = QtGui.QPainter()
        qp.begin(self)
        if self.image:
            qp.drawImage(QtCore.QPoint(0, 0), self.image)
        qp.end()

# QT widget for main window
class MyWindowClass(QtWidgets.QMainWindow, form_class):
    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)
        self.setupUi(self)

        self.startButton.clicked.connect(self.start_clicked)
        self.startDetectionButton.clicked.connect(self.start_detection_clicked)

        self.window_width = self.ImgWidget.frameSize().width()
        self.window_height = self.ImgWidget.frameSize().height()
        self.ImgWidget = OwnImageWidget(self.ImgWidget)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1)

    def start_clicked(self):
        global running
        running = True
        start_thread(capture_thread)
        start_thread(ocr_thread)
        self.startButton.setEnabled(False)
        self.startButton.setText('Starting...')

    def start_detection_clicked(self):
        start_thread(ocr_thread)
        self.startDetectionButton.setEnabled(False)

    def update_frame(self):
        global imageToText
        if not q.empty():

            self.startButton.setText('Camera is live')
            frame = q.get()
            img = frame["img"]
            self.detect_from_image(img)


    def detect_from_image(self, img):
        img = image_processing.pre_process_image(img, self.window_height, self.window_width)
        snack_detection.set_snack_image(img)

        gray = image_processing.process_image_for_ocr(img)

        height, width, bpc = img.shape
        bpl = bpc * width

        employee_detection.set_image_to_text(Image.fromarray(gray))
        image = QtGui.QImage(img.data, width, height, bpl, QtGui.QImage.Format_RGB888)
        self.ImgWidget.setImage(image)

        if (employee_detection.employee_found):
            self.label_2.setText("Employee detected: " + employee_detection.text)
            stop_thread(ocr_thread)
            start_thread(snack_detection_thread)
            employee_detection.set_employee_found(False)

        elif (snack_detection.snack_found):
            self.label_2.setText("snack detected: " + snack_detection.snack)
            stop_thread(snack_detection_thread)
            snack_detection.set_snack_found(False)
            self.startDetectionButton.setEnabled(True)

        else:
            self.label_2.setText("nothing has been found!")


    def closeEvent(self, event):
        global running
        running = False


# thread worker for getting image from webcam
def grab(cam, queue, width, height, fps):
    global running
    capture = cv2.VideoCapture(cam)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)

    while (running):
        frame = {}
        capture.grab()
        retval, img = capture.retrieve(0)
        frame["img"] = img

        if queue.qsize() < 10:
            queue.put(frame)
        else:
            print(queue.qsize())




#setting up threads
capture_thread = threading.Thread(target=grab, args=(0, q, 1920, 1080, 30))
ocr_thread = threading.Thread(target= employee_detection.ocr_worker)
snack_detection_thread = threading.Thread(target= snack_detection.snack_detection_worker)

#starting app
app = QtWidgets.QApplication(sys.argv)
w = MyWindowClass(None)
w.setWindowTitle('Lean Clean Canteen')
w.show()
app.exec_()