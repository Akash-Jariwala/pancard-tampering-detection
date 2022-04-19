from distutils.command.config import config
from pydoc import pager

from cv2 import imread
from numpy import True_, full
from app import app
from flask import request, render_template
import os
import imutils
import cv2
from skimage.metrics import structural_similarity
from PIL import Image
import requests

# adding path to config
app.config['INITIAL_FILE_UPLOAD'] = "app/static/uploads"
app.config['EXISTING_FILE'] = "app/static/original"
app.config['GENERATED_FILE'] = "app/static/generated"

# Route to home page
@app.route('/', method=["GET", "POST"])

def index():

    #  Execute if request is GET
    if request.method == "GET":
        return render_template('index,html')
    
    # Execute if request is POST
    if request.method == "POST":
        # Get uploaded Image
        file_upload = request.files['file_upload']
        filename = file_upload.fliename

        # resize and save the uploaded image
        uploaded_image = Image.open(file_upload).resize((250,160))
        uploaded_image.save(os.path.join(app.config['INITIAL_FILE_UPLOADS'],'image.jpg'))

        # resize and save the original image to ensure that both uploaded are same
        original_image = Image.open(os.path.join(app.config['EXISTING_FILE'], 'image.jpg')).resize((250,160))
        original_image.save(os.path.join(app.config['EXISTING_FILE'], 'image.jpg'))

        # Read uploaded and original image as array
        original_image = imread(os.path.join(app.config['EXISTING_FILE'], 'image.jpg'))
        uploaded_image = imread(os.path.join(app.config['INITIAL_FILE_UPLOADS'], 'image.jpg'))

        # convert image to gray scale
        original_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        uploaded_gray = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)

        # Calculate structural similarity
        (score, diff) = structural_similarity(original_gray, uploaded_gray, full=True)
        diff = (diff*255).astype('uint8')

        # Calculate Threshold and contours
        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # Draw contours on image
        for c in cnts:
            (x,y,w,h) = cv2.boundingRect(c)
            cv2.rectangle(original_image, (x,y), (x+w, y+h), (0,0,255), 2)
            cv2.rectangle(uploaded_image, (x,y), (x+w, y+h), (0,0,255), 2)

        # Save all o/p image if required
        cv2.imwrite(os.path.join(app.config['GENERATED_FILE'], 'image_original.jpg'), original_image)
        cv2.imwrite(os.path.join(app.config['GENERATED_FILE'], 'image_uploaded.jpg'), uploaded_image)
        cv2.imwrite(os.path.join(app.config['GENERATED_FILE'], 'image_diff.jpg'), diff)
        cv2.imwrite(os.path.join(app.config['GENERATED_FILE'], 'image_thresh.jpg'), thresh)
        return render_template('index.html',pred=str(round(score*100,2)) + '%' + ' correct')

    
# Main Function
if __name__ == "__main__":
    app.run(debug=True)
        

