# Tekathon2018

## Setup

To run the program you will need to have python 3.6* installed https://wiki.python.org/moin/BeginnersGuide/Download

1. Clone the repo and cd into the folder

2. install required dependencies 

        pip install -r requirements.txt

3. Install Teseract OCR

    **Windows:**
    
    1. Download Tesseract OCR executables from https://github.com/UB-Mannheim/tesseract/wiki

    2. Add the tesseract directory to your PATH. It is most likely present in ``C:\Program Files (x86)\Tesseract-OCR/tesseract``

    3. Set the environment variable ``TESSDATA_PREFIX`` to the location where folder ``tessdata`` is present. Most probably ``TESSDATA_PREFIX=C:\Program Files (x86)\Tesseract-OCR``.
    
    **Mac:** 
    
        brew install tesseract

6. Run the program with the command 
    
    **Windows:**    
    
        python app.py

    **Mac:**
        
        python3 app.py
    
