# -*- coding: utf-8 -*-
"""
Created on Tue May  5 02:19:04 2020

@author: Heba Gamal El-Din
"""

######################################
""" Importing Necessary Libraries """
#####################################
from pyzbar import pyzbar
import cv2
import datetime

##########################################
""" Life Video Stream Configurations """
#########################################
Cap = cv2.VideoCapture(0)
FPS = Cap.get(cv2.CAP_PROP_FPS)
KPS = 3
hop = round(FPS / KPS)
curr_frame = 0
print(FPS)
csv = open("Output.xlsx", "w")
found = set()

#######################################################################
""" Taking Real Time Frame With Skipping 3 Frames InBetween
    And Decode It By pyzbar Draw Rectangle over Each QR Code Detected
    Then Put The Included Data On Its Location in The Frame 
        Then Save The Resulting Data in An Excel File """
#######################################################################
while True:
    Bool, Frame = Cap.read()
    if not Bool:
        break
    if curr_frame % hop == 0:
        BarCodes = pyzbar.decode(Frame)
        for Code in BarCodes:
            (x, y, w, h) = Code.rect
            cv2.rectangle(Frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            barcodeData = Code.data.decode("utf-8")
            barcodeType = Code.type
            text = "{} ({})".format(barcodeData, barcodeType)
            cv2.putText(Frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            print("[INFO] Found {} barcode: {}".format(barcodeType, barcodeData))
            if barcodeData not in found:
                csv.write("{},{}\n".format(datetime.datetime.now(),
                    barcodeData))
                csv.flush()
                found.add(barcodeData)
        cv2.imshow("QR Code Scanner", Frame)
        if cv2.waitKey(1) == 27:
            break
print("[INFO] cleaning up...")
csv.close()
Cap.release()
cv2.destroyAllWindows()