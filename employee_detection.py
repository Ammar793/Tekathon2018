import pytesseract
import data_storage
import difflib
import re

#contains code in relation to detecting employee
text = "nothing"
image_to_text = 0
employee_names = data_storage.employee_names
def ocr_worker():
    global text
    global image_to_text
    while True:
        print (image_to_text)
        if(image_to_text != 0):
            text = pytesseract.image_to_string(image_to_text)
            print("text")
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
    return proper_name
