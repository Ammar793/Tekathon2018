from PyQt5 import QtCore, QtWidgets, uic, QtGui
import sys
import cv2
import threading
from queue import *
from PIL import Image
import snack_detection
import employee_detection
import image_processing
import math
import winsound



running = False
capture_thread = None
form_class = uic.loadUiType("ui/simple.ui")[0]
q = Queue()
stop_looking_for_employee = False
employee_text= "none"
snack_text= "none"
counter = 0

#main app

#def start_all_threads():
#    capture_thread.start()
#    ocr_thread.start()
#    snack_detection_thread.start()

#def stop_all_threads():
#    capture_thread.stop()
#    ocr_thread.stop()
#    snack_detection_thread.stop()

def start_thread_from_target(thread_target):
    thread = threading.Thread(target=thread_target)
    thread.start()

def start_thread(thread):
    thread.start()

def stop_thread(thread):
    thread.stop_thread()


class CircleWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(CircleWidget, self).__init__(parent)

    def paintEvent(self, event):
        painter = QtGui.QPainter()
        painter.begin(self)
        painter.setPen(QtGui.QPen(QtGui.QColor(230, 110, 110)))
        painter.drawArc(QtCore.QRectF(5, 100, 35, 35), 0, 5760)
        painter.end()


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
            qp.drawImage(QtCore.QPoint(10, 10), self.image)
        qp.end()
loading_counter = 0
loading_showing = False
show_loading = False
# QT widget for main window
class MyWindowClass(QtWidgets.QMainWindow, form_class):
    lobster_font = None
    lobster_font_bold = None
    lobster_font_small = None
    pic = None



    def __init__(self, parent=None):
        global pic
        QtWidgets.QMainWindow.__init__(self, parent)
        self.showFullScreen()
        snack_detection.load_model()

        # tell painter to use your font:
        self.setupUi(self)
        self.set_fonts()

        self.startButton.clicked.connect(self.start_clicked)

        self.startDetectionButton.clicked.connect(self.start_detection_clicked)
        self.reset_button.clicked.connect(self.reset)

        self.window_width = self.ImgWidget.frameSize().width()
        self.window_height = self.ImgWidget.frameSize().height()
        self.ImgWidget = OwnImageWidget(self.ImgWidget)
        #self.circle_widget = CircleWidget(self.circle_widget)
        #self.circle_widget.show()
        #self.circle_widget.show()
        #self.startButton.clicked.connect(self.overlay.show)

        pic = QtWidgets.QLabel(self)
        pic.setPixmap(QtGui.QPixmap("resources/icons/eye2.png"))

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1)

        self.welcome_text.setVisible(False)
        self.employe_info.setVisible(False)
        self.snack_info.setVisible(False)
        self.done_info.setVisible(False)
        self.price_info.setVisible(False)



    def toggle_loading(self, img):

        global loading_counter
        global loading_showing
        global show_loading
        if show_loading and not loading_showing and loading_counter == 700:
            img.show()
            loading_showing= True
            loading_counter = 0
        elif show_loading and loading_showing and loading_counter == 700:
            img.hide()
            loading_showing= False
            loading_counter = 0
        loading_counter = (loading_counter + 1)%701

    def show_loading(self):
        global pic

        if not stop_looking_for_employee:
            pic.move(1300, 250)
            self.toggle_loading(pic)

        elif not snack_detection.stop_looking_for_snack:
            pic.move(1300, 430)
            self.toggle_loading(pic)

        else:
            pic.hide()


    def set_fonts(self):
        global lobster_font
        global lobster_font_bold
        global lobster_font_small
        fontDataBase = QtGui.QFontDatabase()
        font_id = fontDataBase.addApplicationFont("./resources/fonts/Lobster-Regular.ttf")
        families = fontDataBase.applicationFontFamilies(font_id)

        lobster_font = QtGui.QFont(families[0], 26)
        lobster_font_small = QtGui.QFont(families[0], 20)
        lobster_font_bold = QtGui.QFont(families[0], 26, 100)

        font_id2 = fontDataBase.addApplicationFont("./resources/fonts/AvenirLTStd-Light.otf")
        families2 = fontDataBase.applicationFontFamilies(font_id2)
        avenir_font = QtGui.QFont(families2[0], 16)

        self.card_title.setFont(lobster_font_bold)
        self.snack_title.setFont(lobster_font_small)
        self.done_title.setFont(lobster_font_small)
        self.main_title.setFont(lobster_font)
        self.welcome_text.setFont(lobster_font_small)

        self.employe_info.setFont(avenir_font)
        self.snack_info.setFont(avenir_font)
        self.done_info.setFont(avenir_font)
        self.main_info.setFont(avenir_font)
        self.price_info.setFont(avenir_font)

    def reset(self):
        global stop_looking_for_employee
        #ocr_thread = threading.Thread(target=employee_detection.ocr_worker)

        stop_thread(employee_detection)
        stop_thread(snack_detection)

        snack_detection.reset()
        employee_detection.reset()
        start_thread_from_target(employee_detection.ocr_worker)

        if stop_looking_for_employee:

            self.move_element(self.snack_title, -40)
            self.move_element(self.done_title, -40)
            self.move_element(self.snack_info, -40)
            self.move_element(self.done_info, -40)
            self.move_element(self.price_info, -40)

        stop_looking_for_employee = False

        self.card_title.setFont(lobster_font_bold)
        self.snack_title.setFont(lobster_font_small)
        self.done_title.setFont(lobster_font_small)

        self.welcome_text.setVisible(False)
        self.employe_info.setVisible(False)
        self.snack_info.setVisible(False)
        self.done_info.setVisible(False)
        self.price_info.setVisible(False)


    def start_clicked(self):
        global running
        global employee_text
        global snack_text
        running = True
        start_thread(capture_thread)
        self.startButton.setEnabled(False)
        self.startButton.setText('Starting...')
        #self.draw_circle()
        #self.set_text()

    def start_detection_clicked(self):
        global show_loading
        show_loading = True
        start_thread_from_target(employee_detection.ocr_worker)
        self.startDetectionButton.setEnabled(False)

    def update_frame(self):
        global imageToText
        self.show_loading()
        if not q.empty():
            self.startButton.setText('Camera is live')
            frame = q.get()
            img = frame["img"]
            self.detect_from_image(img)

    def set_employee_text(self):
        global lobster_font
        global lobster_font_bold
        global lobster_font_small

        self.employe_info.setText(employee_detection.employee.name)
        self.welcome_text.setVisible(True)
        self.employe_info.setVisible(True)

        self.move_element(self.snack_title, 40)
        self.move_element(self.done_title, 40)
        self.move_element(self.snack_info, 40)
        self.move_element(self.done_info, 40)
        self.move_element(self.price_info, 40)

        self.switchStep(self.card_title, self.snack_title)



    def switchStep(self, currentStep, nexStep):
        currentStep.setFont(lobster_font_small)
        nexStep.setFont(lobster_font_bold)

    def move_element(self, element, move):
        px = element.geometry().x()
        py = element.geometry().y()
        dw = element.width()
        dh = element.height()
        element.setGeometry(px, py + move, dw, dh)


    def set_snack_text(self):
        self.snack_info.setText("Your item is a: " + snack_detection.snack.get_name() )
        self.price_info.setText("Total Today: $" + str(snack_detection.snack.get_price()) + "\n Total Balance: $" + str(employee_detection.employee.get_total()) )
        self.switchStep(self.snack_title, self.done_title)
        self.snack_info.setVisible(True)
        self.done_info.setVisible(True)
        self.price_info.setVisible(True)


    def play_sound(self):
        winsound.PlaySound('./resources/sound/beep.wav', winsound.SND_FILENAME)

    def detect_from_image(self, img):
        global counter
        global stop_looking_for_employee
        img = image_processing.pre_process_image(img, self.window_height, self.window_width)

        snack_imag = image_processing.process_image_for_snack_detection(img)
        filename = "choc_" + str(counter)+".jpg"
        counter = counter + 1
        cv2.imwrite("C:/Users/mammar/PycharmProjects/Hackathon/training/images/chocs/"+filename , snack_imag)
        snack_detection.set_snack_image(snack_imag)

        gray = image_processing.process_image_for_ocr(img)

        height, width, bpc = img.shape
        bpl = bpc * width

        employee_detection.set_image_to_text(Image.fromarray(gray))
        image = QtGui.QImage( image_processing.make_rectangle(img).data, width, height, bpl, QtGui.QImage.Format_RGB888)
        self.ImgWidget.setImage(image)

        if (employee_detection.employee_found and not stop_looking_for_employee):


            self.set_employee_text()
            #stop_thread(ocr_thread)
            snack_detection.stop_looking_for_snack = False
            start_thread_from_target(snack_detection.snack_detection_worker)
            #employee_detection.set_employee_found(False)
            self.play_sound()
            stop_looking_for_employee = True


        elif (snack_detection.snack_found and not snack_detection.stop_looking_for_snack):
            employee_detection.employee.add_to_total(snack_detection.snack.get_price())
            self.set_snack_text()
            self.play_sound()
            snack_detection.stop_looking_for_snack = True
            #stop_thread(snack_detection_thread)
            #snack_detection.set_snack_found(False)
            #self.startDetectionButton.setEnabled(True)

        #else:
        #    self.label_2.setText("nothing has been found!")


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
#ocr_thread = threading.Thread(target= employee_detection.ocr_worker)
#snack_detection_thread = threading.Thread(target= snack_detection.snack_detection_worker)

#starting app
app = QtWidgets.QApplication(sys.argv)
w = MyWindowClass(None)
w.setWindowTitle('Lean Clean Canteen')
w.show()
app.exec_()