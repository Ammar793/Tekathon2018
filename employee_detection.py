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
employees_found = []


class employee_detection_thread(threading.Thread):
    #contains code in relation to detecting employee

    def __init__(self):
        self._stopevent = threading.Event()
        self._sleepperiod = 0.2
        threading.Thread.__init__(self)

    def run(self):
        global try_counter
        global image_to_text
        global cv_image

        while not self._stopevent.isSet( ):
            print(image_to_text)
            if image_to_text != 0:
                if segmentation.check_if_item_present(cv_image):
                    try_counter += 1
                    set_item_present(True)

                    text = pytesseract.image_to_string(image_to_text)
                    print(text)
                    text = text.replace('\n', ' ').replace('\r', '')
                    text = re.sub(' +', ' ', text)
                    set_employee(find_name_from_database(text))
                else:
                    try_counter = 0
                    set_item_present(False)
            self._stopevent.wait(self._sleepperiod)

    def join(self, timeout=None):
        """ Stop the thread and wait for it to end. """
        self._stopevent.set( )
        threading.Thread.join(self, timeout)


def set_item_present(boolean):
    global item_present
    item_present = boolean


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


def set_employee(employee_object):
    global employee
    employee = employee_object


def set_employee_from_name(name):
    global employee
    global employees_found

    employee_list = [x for x in employees_found if x.name == name]
    if len(employee_list) > 0:
        successfully_found()
        employee = employee_list[0]
    else:
        employee_name = [x for x in employees if x == name][0]
        employee_temp = data_storage.Employee(employee_name)
        employee = employee_temp
        employees_found.append(employee_temp)


def find_name_from_database(text):
    global employees_found
    split_text = text.split(" ")
    proper_name = text
    new_name = ""
    for name in split_text:
        new_name = new_name + name + " "
        matches = difflib.get_close_matches(proper_name, employee_names)
        if( len(matches) >0):
            proper_name = matches[0]
            employee = [x for x in employees_found if x.name == proper_name]
            if(len(employee)>0):
                successfully_found()
                return employee[0]
            else:
                employee = [x for x in employees if x == proper_name][0]
                employee_obj = data_storage.Employee(employee)
                successfully_found()
                employees_found.append(employee_obj)
                return employee_obj
    return data_storage.Employee(proper_name)


