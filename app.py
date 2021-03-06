from PyQt5 import QtCore, QtWidgets, uic, QtGui
import sys
import cv2
import threading
from queue import *
from PIL import Image
from visual_cortex import employee_detection, image_processing, snack_detection
from memory import data_storage
from ui import sound_effects

running = False
capture_thread = None
form_class = uic.loadUiType("ui/simple.ui")[0]
q = Queue()
stop_looking_for_employee = False
big_square = False
today_total = 0

employee_detection_thread = None
snack_detection_thread = None
loading_counter = 0
loading_showing = False
show_loading = False

lobster_font = None
lobster_font_bold = None
lobster_font_small = None


def start_thread(thread):
    thread.start()


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


# QT widget for main window
class MyWindowClass(QtWidgets.QMainWindow, form_class):
    pic = None

    def __init__(self, parent=None):
        global pic
        global snack_detection_thread
        global employee_detection_thread
        QtWidgets.QMainWindow.__init__(self, parent)
        self.showFullScreen()
        snack_detection.load_model()

        snack_detection_thread = snack_detection.snack_detection_thread()
        employee_detection_thread = employee_detection.employee_detection_thread()

        # tell painter to use your font:
        self.setupUi(self)
        self.set_fonts()
        self.setup_buttons()

        self.window_width = self.ImgWidget.frameSize().width()
        self.window_height = self.ImgWidget.frameSize().height()
        self.ImgWidget = OwnImageWidget(self.ImgWidget)

        pic = QtWidgets.QLabel(self)
        pic.setPixmap(QtGui.QPixmap("resources/icons/eye2.png"))

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(1)

        self.setup_dropdown()
        self.hide_elements_initial()

    def setup_buttons(self):
        self.startButton.clicked.connect(self.start_clicked)
        self.startDetectionButton.clicked.connect(self.start_detection_clicked)
        self.reset_button.clicked.connect(self.reset)
        self.done_button.clicked.connect(self.reset)
        self.add_button.clicked.connect(self.add_item)

    def hide_elements_initial(self):
        self.welcome_text.setVisible(False)
        self.employe_info.setVisible(False)
        self.snack_info.setVisible(False)
        self.done_info.setVisible(False)
        self.today_title.setVisible(False)
        self.total_title.setVisible(False)
        self.today_info.setVisible(False)
        self.total_info.setVisible(False)
        self.done_button.setVisible(False)
        self.snack_subtitle.setVisible(False)
        self.name_list.setVisible(False)
        self.sorry_text.setVisible(False)
        self.add_button.setVisible(False)

    def setup_dropdown(self):
        emp_names = data_storage.get_employee_names_list()
        self.name_list.addItem("")
        for name in emp_names:
           self.name_list.addItem(name)

        self.name_list.currentIndexChanged.connect(self.name_selected)

    def toggle_loading(self, img):
        global loading_counter
        global loading_showing
        global show_loading

        interval_length = 200

        if show_loading and not loading_showing and loading_counter == interval_length:
            img.show()
            loading_showing= True
            loading_counter = 0
        elif show_loading and loading_showing and loading_counter == interval_length:
            img.hide()
            loading_showing= False
            loading_counter = 0
        loading_counter = (loading_counter + 1) % interval_length+1

    def show_loading(self):
        global pic
        if not stop_looking_for_employee and employee_detection.item_present:
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
        lobster_font_smallest = QtGui.QFont(families[0], 16)
        lobster_font_bold = QtGui.QFont(families[0], 26, 100)

        font_id2 = fontDataBase.addApplicationFont("./resources/fonts/AvenirLTStd-Light.otf")
        families2 = fontDataBase.applicationFontFamilies(font_id2)
        avenir_font = QtGui.QFont(families2[0], 16)
        avenir_font_big = QtGui.QFont(families2[0], 26, 200)
        avenir_font_medium = QtGui.QFont(families2[0], 24, 100)

        self.card_title.setFont(lobster_font_bold)
        self.snack_title.setFont(lobster_font_small)
        self.done_title.setFont(lobster_font_small)
        self.main_title.setFont(lobster_font)
        self.welcome_text.setFont(lobster_font_small)
        self.today_title.setFont(lobster_font_smallest)
        self.total_title.setFont(lobster_font_smallest)

        self.employe_info.setFont(avenir_font)
        self.snack_info.setFont(avenir_font_big)
        self.done_info.setFont(avenir_font)
        #self.main_info.setFont(avenir_font)
        self.today_info.setFont(avenir_font_medium)
        self.total_info.setFont(avenir_font_medium)
        self.snack_subtitle.setFont(avenir_font)
        self.sorry_text.setFont(lobster_font_small)

    def reset(self):
        global stop_looking_for_employee
        global employee_detection_thread
        global snack_detection_thread
        global big_square
        global today_total
        #ocr_thread = threading.Thread(target=employee_detection.ocr_worker)

        today_total = 0
        employee_detection_thread.join()

        if big_square:
            snack_detection_thread.join()

        snack_detection_thread = snack_detection.snack_detection_thread()
        employee_detection_thread = employee_detection.employee_detection_thread()
        snack_detection.reset()
        employee_detection.reset()
        start_thread(employee_detection_thread)

        if stop_looking_for_employee:
            self.move_second_step_elements(-40)
            #self.move_element(self.price_info, -40)

        stop_looking_for_employee = False

        self.card_title.setFont(lobster_font_bold)
        self.snack_title.setFont(lobster_font_small)
        self.done_title.setFont(lobster_font_small)
        self.hide_elements_initial()

        big_square = False

    def start_clicked(self):
        global running
        running = True
        start_thread(capture_thread)
        self.startButton.setEnabled(False)
        self.startButton.setText('Starting...')
        #self.draw_circle()
        #self.set_text()

    def start_detection_clicked(self):
        global show_loading
        global employee_detection_thread
        show_loading = True
        employee_detection_thread = employee_detection.employee_detection_thread()
        start_thread(employee_detection_thread)
        self.startDetectionButton.setEnabled(False)

    def update_frame(self):
        self.show_loading()
        if not q.empty():
            self.startButton.setText('Camera is live')
            frame = q.get()
            img = frame["img"]
            self.detect_from_image(img)

    def move_second_step_elements(self, value):
        self.move_element(self.snack_title, value)
        self.move_element(self.done_title, value)
        self.move_element(self.snack_info, value)
        self.move_element(self.done_info, value)
        self.move_element(self.snack_subtitle, value)

    def set_employee_text(self, moved):
        global lobster_font
        global lobster_font_bold
        global lobster_font_small

        self.employe_info.setText(employee_detection.employee.name)
        self.welcome_text.setVisible(True)
        self.employe_info.setVisible(True)
        if not moved:
            self.move_second_step_elements(40)
       # self.move_element(self.price_info, 40)
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
        global today_total

        today_total = today_total + snack_detection.snack.get_price()

        self.snack_info.setText(snack_detection.snack.get_name())
        self.today_info.setText("$" + str(today_total))
        self.total_info.setText("$" + str(employee_detection.employee.get_total()))

        self.switchStep(self.snack_title, self.done_title)
        self.snack_info.setVisible(True)
        self.done_info.setVisible(True)
        self.today_title.setVisible(True)
        self.total_title.setVisible(True)
        self.today_info.setVisible(True)
        self.total_info.setVisible(True)
        self.done_button.setVisible(True)
        self.snack_subtitle.setVisible(True)
        self.add_button.setVisible(True)

    def show_dropdown(self):
        line_edit = QtWidgets.QLineEdit()
        line_edit.setPlaceholderText("Please type your name then press Enter")
        self.name_list.setCurrentIndex(0)
        self.name_list.setLineEdit(line_edit)

        self.move_element(self.snack_title, 40)
        self.move_element(self.done_title, 40)
        self.move_element(self.snack_info, 40)
        self.move_element(self.done_info, 40)
        self.move_element(self.snack_subtitle, 40)

        self.sorry_text.setVisible(True)
        self.name_list.setVisible(True)

    def name_selected(self, i):
        global big_square
        global snack_detection_thread
        global employee_detection_thread

        if i == 0:
            return

        big_square = True
        employee_detection.set_employee_from_name(self.name_list.currentText())
        self.set_employee_text(True)

        self.name_list.setVisible(False)
        self.sorry_text.setVisible(False)
        snack_detection.stop_looking_for_snack = False

        employee_detection_thread.join()
        start_thread(snack_detection_thread)


    def unset_snack_text(self):
        self.snack_info.setText("")
        self.today_info.setText("")
        self.total_info.setText("")

        self.switchStep(self.done_title, self.snack_title)
        self.snack_info.setVisible(False)
        self.done_info.setVisible(False)
        self.today_title.setVisible(False)
        self.total_title.setVisible(False)
        self.today_info.setVisible(False)
        self.total_info.setVisible(False)
        self.done_button.setVisible(False)
        self.add_button.setVisible(False)

    def add_item(self):
        snack_detection.stop_looking_for_snack = False
        self.unset_snack_text()

    def detect_from_image(self, img):
        global stop_looking_for_employee
        global big_square

        img = image_processing.pre_process_image(img, self.window_height, self.window_width)

        crop_settings = image_processing.crop_values_card
        if(big_square):
            crop_settings = image_processing.crop_values

        img_init = image_processing.process_image_for_presence_check(img, crop_settings)
        snack_imag = image_processing.process_image_for_snack_detection(img)
        snack_detection.set_snack_image(snack_imag)

        gray = image_processing.process_image_for_ocr(img)

        height, width, bpc = img.shape
        bpl = bpc * width

        employee_detection.set_image_to_text(Image.fromarray(gray))
        employee_detection.set_cv_image(img_init)

        image = QtGui.QImage(image_processing.make_rectangle(img, crop_settings).data, width, height, bpl, QtGui.QImage.Format_RGB888)
        self.ImgWidget.setImage(image)

        number_of_tries = employee_detection.try_counter

        if number_of_tries > 15 and not stop_looking_for_employee:
            self.show_dropdown()
            stop_looking_for_employee = True

        if employee_detection.employee_found and not stop_looking_for_employee:
            self.set_employee_text(False)
            snack_detection.stop_looking_for_snack = False
            big_square = True

            sound_effects.play_sound()
            stop_looking_for_employee = True

            employee_detection_thread.join()
            start_thread(snack_detection_thread)

        elif snack_detection.snack_found and not snack_detection.stop_looking_for_snack:
            employee_detection.employee.add_to_total(snack_detection.snack.get_price())
            self.set_snack_text()
            sound_effects.play_sound()
            snack_detection.stop_looking_for_snack = True

    def closeEvent(self, event):
        global running
        running = False

#thread worker for getting image from webcam
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


capture_thread = threading.Thread(target=grab, args=(0, q, 1920, 1080, 30))
app = QtWidgets.QApplication(sys.argv)
w = MyWindowClass(None)
w.setWindowTitle('Lean Clean Canteen')
w.show()
app.exec_()