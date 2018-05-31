import pytesseract


#contains code in relation to detecting employee

image_to_text = 0
def ocr_worker():
    global image_to_text
    while True:
        print (image_to_text)
        if(image_to_text != 0):
            text = pytesseract.image_to_string(image_to_text)
            print("text")
            print(text)
