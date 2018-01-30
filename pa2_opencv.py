# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import cv2                            
import numpy as np                           #importing libraries

    
def mySkin(img):
    #print img
    if (img is not None):
        img_out = np.zeros((img.shape[0],img.shape[1])).astype('uint8')
    
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                B = img[i][j][0]
                G = img[i][j][1]
                R = img[i][j][2]
                if ((R > 95 and G > 40 and B > 20) and (max(R, G, B) - min(R, G, B) > 15) and (abs(R - G) > 15) and (R > G) and (R > B)):
                    img_out[i][j] = 255
        return img_out
    
def myDiff(prev, curr):
    
    dif = cv2.absdiff(prev,curr)
    #gs = hsv(dif)
    gs = cv2.cvtColor(dif,cv2.COLOR_BGR2GRAY)
    #print gs
    dst = gs>50
    #print dst
    dst1 = dst.astype('uint8')
    dst1 *= 255
    return dst1


def myMot(im, im1, im2):
    
    #img_out = np.zeros((im.shape[0],im.shape[1])).astype('uint8')
#    for i in range(img.shape[0]):
#        for j in range(img.shape[1]):
#            #print im[i][j]
#            if (im[i][j] == 255 or im1[i][j] == 255 or im2[i][j] == 255):
#                img_out[i][j] = 255
                        
    im_sum = np.add(im2, np.add(im, im1))
    im_bool = im_sum >=255
    im_val = im_bool.astype('uint8')
    im_val *= 255
    return im_val


def hsv(img):
    
    imag = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return cv2.inRange(imag, (0,100,50), (50,150,180))


def close(img):
    
    one = np.ones((5,5))
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, one)
    return closing


def dil(img):
    
    one = np.ones((5,5))
    dila = cv2.dilate(img, one)
    return dila


def con(img):
    
    _, contours, _ = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return np.array([[0,0],[0,1], [1,1], [1,0]])
    max_area = 0
    ci = 0
    for i in range(len(contours)):
            cnt=contours[i]
            area = cv2.contourArea(cnt)
            if(area>max_area):
                max_area=area
                ci=i
    #print contours[ci]
    return contours[ci]


def in_ang(x1,y1,x2,y2,cx,cy):
    
    dis = np.sqrt((x1-cx)**2 + (y1-cy)**2)
    dis2 = np.sqrt((x2-cx)**2 + (y2-cy)**2)
    if(dis < dis2):
        Ax = x2
        Ay = y2
        Bx = x1
        By = y1
    else:
        Ax = x1
        Ay = y1
        Bx = x2
        By = y2
        
    val0 = cx - Ax
    val1 = cy - Ay
    val2 = Bx - Ax
    val3 = By - Ay
    
    return (np.arccos((val0*val2 + val1*val3)/ (np.sqrt(val2**2 + val3**2) * np.sqrt(val0**2 + val1**2))) * 180)/np.pi
    



def defec(cnt, hull, img, img2):
    defects = cv2.convexityDefects(cnt,hull)
    rx,ry,rw,rh = cv2.boundingRect(cnt)
    cen = [(rx + rw) / 2, (ry + rh) / 2]
    points = []
    if(defects is not None):
        for i in range(defects.shape[0]):
            s,e,f,d = defects[i,0]
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])
            ang = (np.arctan2(cen[1] - start[1], cen[0] - start[0])*180)/(np.pi)
            inang = in_ang(start[0],start[1],end[0],end[1],far[0],far[1])
            leng = np.sqrt((start[0] - far[0])**2 + (start[1] - far[1])**2)
            cv2.line(img,start,far,[255,0,0],2)                   
            cv2.circle(img,far,5,[0,0,255],-1)
            #print ang
            if (ang > -160 and ang < 160 and abs(inang) > 20 and abs(inang) < 120 and leng > 0.1 * rh):
                points += [start]
        #print(len(points))
        #print rw
        if len(points) > 3 and rw < 130 and rw > 90:
            cv2.putText(img2, "Give Me Five!", (cen[0]+25, cen[1]+50), cv2.FONT_HERSHEY_SIMPLEX, 1, [255,255,0],3, cv2.LINE_AA)
            #print "Hello World!"
        elif len(points) <3 and rw > 50 and rw < 100:
            cv2.putText(img2, "Pound It Bro!", (cen[0]+25, cen[1]+50), cv2.FONT_HERSHEY_SIMPLEX, 1, [255,255,0],3, cv2.LINE_AA)
            #print "Fist!"
        elif len(points) > 3 and rw > 150:
            cv2.putText(img2, "Hey There Good Looking!", (cen[0]+25, cen[1]+50), cv2.FONT_HERSHEY_SIMPLEX, 1, [255,255,0],3, cv2.LINE_AA)
            #print "Waving!"
            
        for i in range(len(points)):
            cv2.circle(img,points[i], 9,[0,0,255],-1)
            #dist = cv2.pointPolygonTest(cnt,centr,True)

      
        
delay = 4
cap = cv2.VideoCapture(0)                #creating camera object
imgs = np.zeros((delay,480,640,3))
#print imgs
for i in range(delay):
    _,img_temp = cap.read()
    imgs[i] = img_temp
#ret0,img0 = cap.read()
#print(img0.shape)
#ret1,img1 = cap.read()
#ret2,img2 = cap.read()
#img1grey = myDiff(img0, img1)
#img2grey = myDiff(img1, img2)
while( cap.isOpened() ) :
    ret,img = cap.read()  
    #if (img is None): print img                       #reading the frame
    ################################################
#    md = myDiff(img2,img)
#    mot = myMot(md, img1grey, img2grey)
    ################################################
#    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#    grey = mySkin(img)
#    blur = cv2.GaussianBlur(grey,(5,5),0)
#    ret,thresh1 = cv2.threshold(blur,128,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ################################################
    hue = hsv(img)
    hue_all = hue
    for i in range(delay):
        #print imgs[1]
        hue_all = np.add(hue_all, hsv(imgs[i].astype('uint8')))
        hue_all = hue_all >= 255
        hue_all = hue_all.astype('uint8')
        hue_all *= 255
        
    
#    hue_blur = cv2.GaussianBlur(hue,(5,5),0)
#    hue_blur2 = cv2.medianBlur(hue, 5)
#    hue_blur3 = cv2.bilateralFilter(hue, 5, 10,3)
    
    ################################################
    cnt = con(dil(hue_all))
    hull = cv2.convexHull(cnt)
    drawing = np.zeros(img.shape,np.uint8)
    cv2.drawContours(drawing,[cnt],0,(0,255,0),2)
    cv2.drawContours(drawing,[hull],0,(0,0,255),2)
    ################################################
    hull = cv2.convexHull(cnt,returnPoints = False)
    defec(cnt, hull, drawing, img)
    #cv2.putText(img, "That's a Fist!", (0, img.shape[0]-1), cv2.FONT_HERSHEY_SIMPLEX, 4, [255,0,0],2, cv2.LINE_AA)
    ################################################
    cv2.imshow('draw',drawing)                  #displaying the frames
#    cv2.imshow('hue', dil(hue))
#    cv2.imshow('hue_all', dil(hue_all))
    cv2.imshow('image', img)
#    cv2.imshow('nat', img)
#    cv2.imshow('hue_blur2', dil(hue))
#    cv2.imshow('hue_blur3', hue_blur3)
    ################################################
    imgs2 = np.zeros((10,480,640,3))
    imgs2[0] = img
    for i in range(delay-1):
        #print imgs2.shape, imgs[i].shape
        imgs2[i+1] = imgs[i]
    imgs = imgs2
#    img0 = img1
#    img1 = img2
#    img2 = img
#    img1grey = img2grey
#    img2grey = md
    ################################################
    k = cv2.waitKey(10)
    if k == 27:
        break
cap.release()