import pytesseract
import data_storage
import difflib
import re
import segmentation
import threading

employee = "nothing"
image_to_text = 0
cv_image = 0
employees = data_storage.employees
employee_names = data_storage.get_employee_names_list()
employee_found = False
item_present = False
try_counter = 0

class employee_detection_thread(threading.Thread):

    #contains code in relation to detecting employee

    def __init__(self):
        self._stopevent = threading.Event()
        self._sleepperiod = 0.2
        threading.Thread.__init__(self)

    def run(self):
        global try_counter
        global employee_found
        global employee
        global image_to_text
        global cv_image
        global item_present
        while not self._stopevent.isSet( ):
            print(image_to_text)
            if (image_to_text != 0):
                if (segmentation.check_if_item_present(cv_image)):
                    try_counter += 1
                    item_present = True
                    text = pytesseract.image_to_string(image_to_text)
                    print(text)
                    text = text.replace('\n', ' ').replace('\r', '')
                    text = re.sub(' +', ' ', text)
                    employee = check_if_two_names(text)
                else:
                    try_counter = 0
                    item_present = False
            self._stopevent.wait(self._sleepperiod)

    def join(self, timeout=None):
        """ Stop the thread and wait for it to end. """
        self._stopevent.set( )
        threading.Thread.join(self, timeout)

def set_employee_found(boolean):
    global employee_found
    employee_found = boolean

def set_image_to_text(image_array):
    global image_to_text
    image_to_text = image_array


def set_cv_image(image_array):
    global cv_image
    cv_image = image_array



def reset():
    set_employee_found(False)
    set_image_to_text(0)
    set_try_counter(0)

def set_try_counter(counter):
    global try_counter
    try_counter = counter

def successfully_found():
    set_employee_found(True)

def ocr_worker():
    global try_counter
    global employee_found
    global employee
    global image_to_text
    global cv_image
    global item_present
    while not employee_found:
        print (image_to_text)
        if(image_to_text != 0):
            if(segmentation.check_if_item_present(cv_image)):
                try_counter+=1
                item_present = True
                text = pytesseract.image_to_string(image_to_text)
                print(text)
                text = text.replace('\n', ' ').replace('\r', '')
                text = re.sub(' +', ' ', text)
                employee = check_if_two_names(text)
            else:
                try_counter = 0
                item_present = False

            #print( )

def check_if_two_names(text):
    split_text = text.split(" ")
    proper_name = text
    new_name = ""
    for name in split_text:
        new_name = new_name + name + " "
        matches = difflib.get_close_matches(proper_name, employee_names)
        if( len(matches) >0):
            proper_name = matches[0]
            employee = [x for x in employees if x.name == proper_name][0]
            successfully_found()
            return employee
    return data_storage.Employee(proper_name)

def set_employee(name):
    global employee
    employee = [x for x in employees if x.name == name][0]


