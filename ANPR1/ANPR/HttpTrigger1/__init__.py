import logging
import imutils
import numpy as np
import pytesseract
import cv2
from datetime import datetime
import tempfile
import pandas as pd
import re, os
import tempfile
import azure.functions as func
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, __version__


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    # name = req.params.get('name')
    print("ok")
    cwd = os.getcwd()
    print("++++++++print++++++++" ,str(cwd), "++++++++print+++++++")
    # logging.info("++++++++++++++++" ,str(cwd), "+++++++++++++++")
    print(os.listdir(str(cwd)))
    # logging.info("++++++++++++++++" ,os.listdir(str(cwd)), "+++++++++++++++")
    # import subprocess
    from subprocess import STDOUT, check_call
    # check_call(['apt-get', 'install', '-y', 'apt-transport-https'])
    # check_call(['apt-get', 'install', '-y', 'tesseract-ocr'])

    # cmd = "D:/Official/non-confidential/ANPR/img_devops/ANPRimage/tesseract/tesseract.exe" #"/home/site/wwwroot/tesseract/tesseract.exe"

    # returned_value = subprocess.call(cmd, shell=True)  # returns the exit code in unix
    # print('returned value:' + str(returned_value))
    # logging.debug('returned value:' + str(returned_value))
    loc = find_tesseract("tesseract.exe","/home/")
    # pytesseract.pytesseract.tesseract_cmd = loc[0]
    connect_str = os.getenv("connect_str")

    # Create the BlobServiceClient object which will be used to create a container client
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    images_container_name = "rawdatastore/anprimages"
    loaded_imgs = []
    labels = []

    results = pd.DataFrame(columns = ["image_name", "label"])

    img_names = get_img_names(connect_str)

    for img in img_names:
        tempFilePath = tempfile.mkdtemp()
        local_file_name = img
        local_path = tempFilePath

        blob_client = blob_service_client.get_blob_client(container=images_container_name, blob=local_file_name)

        download_file_path = os.path.join(local_path,local_file_name) 
    #     print("\nDownloading blob to \n\t" + download_file_path)

        with open(download_file_path, "wb") as download_file:
            download_file.write(blob_client.download_blob().readall())
            
        imgs = cv2.imread(download_file_path,cv2.IMREAD_COLOR)
        
        loaded_imgs.append(imgs)
        #label the images
        labels.append(label_images(imgs, img, connect_str))
        
        #save labels in csv file
        results["image_name"] = img_names
        results["label"] = labels

        ## saving csv file to datastore
        tempFilePath = tempfile.mkdtemp()

        local_file_name = "results.csv"
        logging.info('Going to write in temp folder')
        results.to_csv(tempFilePath + "/" + local_file_name, index=False)
        logging.debug('File written to temp storage')
        upload_file_path = tempFilePath + "/" + local_file_name
        # below connection string of other resource group can also be used
        blob_service_client = BlobServiceClient.from_connection_string(connect_str)
        print('Connected with Azure storage!')
        logging.info('Connected with Azure storage!')
        # Create a unique name for the container
        container_name = "publishdatastore/anprimages"

        try:
            blob_client = blob_service_client.get_blob_client(container=container_name, blob=local_file_name)

            logging.info("\nUploading to Azure Storage as blob:\n\t" + local_file_name)

            # Upload the created file
            with open(upload_file_path, "rb") as data:
                blob_client.upload_blob(data, overwrite=True)

            logging.info('Completed!')

        except Exception:
            #handle any missing key errors
            print('Some error occurred while uploading the output file!!')




    # if not name:
    #     try:
    #         req_body = req.get_json()
    #     except ValueError:
    #         pass
    #     else:
    #         name = req_body.get('name')

    # if name:
    #     return func.HttpResponse(f"Hello, {name}. This HTTP triggered function executed successfully.")
    # else:
    #     return func.HttpResponse(
    #          "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
    #          status_code=200
    #     )


def label_images(img, img_name, connect_str):
#     img = cv2.imread(img,cv2.IMREAD_COLOR)
    img = cv2.resize(img, (600,400) )

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    gray = cv2.bilateralFilter(gray, 13, 15, 15) 

    edged = cv2.Canny(gray, 30, 200) 
    contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
    screenCnt = None

    for c in contours:

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)

        if len(approx) == 4:
            screenCnt = approx
            break

    if screenCnt is None:
        detected = 0
        print ("No contour detected")
    else:
         detected = 1

    if detected == 1:
        cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)

    mask = np.zeros(gray.shape,np.uint8)
    new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
    new_image = cv2.bitwise_and(img,img,mask=mask)
    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    Cropped = gray[topx:bottomx+1, topy:bottomy+1]


    text = pytesseract.image_to_string(Cropped, config='--psm 11') 
    text = re.sub('[\W_]', '', text)
    font = cv2.FONT_HERSHEY_SIMPLEX
    (x, y - 5)
    res = cv2.putText(img, text=text, org=(topx, topy), fontFace=font, fontScale=1, color=(255,0,0), thickness=2, lineType=cv2.LINE_AA)

    print("Detected license plate Number is:",text)
    img = cv2.resize(img,(500,300))
    Cropped = cv2.resize(Cropped,(400,200))

    status = save_data_to_datastore(img_name, img, connect_str)
    print(img_name + '_op.jpg', status)
    return text

def save_data_to_datastore(file_name, output_img, connect_str):
    x = datetime.today()
    tempFilePath = tempfile.mkdtemp()

    local_file_name = file_name.split(".")[0] + '_op.jpg'
    logging.info('Going to write in temp folder')
    cv2.imwrite(tempFilePath + "/" +local_file_name,output_img)
    logging.debug('File written to temp storage')
    upload_file_path = tempFilePath + "/" + local_file_name
    # below connection string of other resource group can also be used
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)
    print('Connected with Azure storage!')
    logging.info('Connected with Azure storage!')
    # Create a unique name for the container
    container_name = "publishdatastore/anprimages"

    try:
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=local_file_name)

        logging.info("\nUploading to Azure Storage as blob:\n\t" + local_file_name)

        # Upload the created file
        with open(upload_file_path, "rb") as data:
            blob_client.upload_blob(data, overwrite=True)

        logging.info('Completed!')

        y = datetime.today()
        diff = (y-x).seconds/60
        logging.info('Data saved to datastore in ' + str(round(diff,2)) + ' mins')


    except Exception:
        #handle any missing key errors
        print('Some error occurred while uploading the output file!!')
        return "Failed !!"

    return "Successful!"

def get_img_names(connect_str):
    container = ContainerClient.from_connection_string(connect_str, container_name="rawdatastore")
    blob_list = container.list_blobs()
    blobs = [i['name'] for i in blob_list]
    img_names = []
    
    for item in blobs:
        items = item.split("/")
        if len(items) >= 2 and items[0] == 'anprimages':
            img_names.append(items[1])
    print(len(img_names)) 
    return img_names

def find_tesseract(filename, search_path):
    result = []

    # Wlaking top-down from the root
    for root, dir, files in os.walk(search_path):
        if filename in files:
            result.append(os.path.join(root, filename))
    return result


# main("329")