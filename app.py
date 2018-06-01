from PyQt5 import QtCore, QtWidgets, uic, QtGui
import sys
import cv2
import threading
from queue import *
from PIL import Image
import snack_detection
import employee_detection

running = False
capture_thread = None
form_class = uic.loadUiType("ui/simple.ui")[0]
q = Queue()


#main app


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

        self.window_width = self.ImgWidget.frameSize().width()
        self.window_height = self.ImgWidget.frameSize().height()
        self.ImgWidget = OwnImageWidget(self.ImgWidget)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1)

    def start_clicked(self):
        global running
        running = True
        capture_thread.start()
        ocrThread.start()
        #snackDetectionThread.start()
        self.startButton.setEnabled(False)
        self.startButton.setText('Starting...')


    def update_frame(self):
        if not q.empty():
            self.startButton.setText('Camera is live')
            frame = q.get()
            img = frame["img"]

            img_height, img_width, img_colors = img.shape
            scale_w = float(self.window_width) / float(img_width)
            scale_h = float(self.window_height) / float(img_height)
            scale = min([scale_w, scale_h])

            if scale == 0:
                scale = 1

            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

            gray = cv2.threshold(gray, 100, 255,
                                 cv2.THRESH_BINARY)[1]

            #gray = cv2.medianBlur(gray,3)
            #cv2.imshow("grey",gray)

            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
           # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

           # gray = np.float32(gray)
            #dst = cv2.cornerHarris(gray, 2, 3, 0.04)

           # edges = cv2.Canny(img, 100, 200)

            # result is dilated for marking the corners, not important
            #dst = cv2.dilate(dst, None)

            # Threshold for an optimal value, it may vary depending on the image.
            #img[dst > 0.01 * dst.max()] = [0, 0, 255]

            #filename = "{}.png".format(os.getpid())
            #cv2.imwrite(filename, gray)
            global imageToText

            snack_detection.snack_image = img
            employee_detection.image_to_text = Image.fromarray(gray)

            #print(text)
            #with PyTessBaseAPI() as api:
             #   api.SetImageFile(Image.fromarray(img))
              #  print(api.GetUTF8Text())
               # print(api.AllWordConfidences())


            #height, width  = edges.shape
            height, width, bpc = img.shape
            bpl = bpc * width
            image = QtGui.QImage(img.data, width, height, bpl, QtGui.QImage.Format_RGB888 )
            self.ImgWidget.setImage(image)
            self.label_2.setText(employee_detection.text)

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
ocrThread = threading.Thread(target= employee_detection.ocr_worker)
snackDetectionThread = threading.Thread(target= snack_detection.snack_detection_worker)


#starting app
app = QtWidgets.QApplication(sys.argv)
w = MyWindowClass(None)
w.setWindowTitle('Lean Clean Canteen')
w.show()
app.exec_()