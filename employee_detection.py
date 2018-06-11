import pytesseract
import data_storage
import difflib
import re

#contains code in relation to detecting employee
text = "nothing"
image_to_text = 0
employee_names = data_storage.employee_names
employee_found = False

def set_employee_found(boolean):
    global employee_found
    employee_found = boolean

def set_image_to_text(image_array):
    global image_to_text
    image_to_text = image_array

def reset():
    set_employee_found(False)
    set_image_to_text(0)

def successfully_found():
    set_employee_found(True)

def ocr_worker():
    global employee_found
    global text
    global image_to_text
    while not employee_found:
        print (image_to_text)
        if(image_to_text != 0):
            text = pytesseract.image_to_string(image_to_text)
            print(text)
            text = text.replace('\n', ' ').replace('\r', '')
            text = re.sub(' +', ' ', text)
            text = check_if_two_names(text)
            #print( )

def check_if_two_names(text):
    split_text = text.split(" ")
    proper_name = text
    if(len(split_text) ==2):
        proper_name = ""
        proper_name = proper_name + split_text[1]
        proper_name = proper_name + ", "
        proper_name = proper_name + split_text[0]
        matches = difflib.get_close_matches(proper_name, employee_names)
        if( len(matches) >0):
            proper_name = matches[0]
            successfully_found()
    return proper_name

def stop_thread():
    global employee_found
    employee_found = True