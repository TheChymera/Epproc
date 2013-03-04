#!/usr/bin/env python
from __future__ import division
__author__ = 'Horea Christian'
from os import path, listdir
from funct import Load, mk_ffile
import cv2
import numpy as np

#INPUT
globalpath = '~/Data/face-pictures/opencv/'
folder_in = 'in/'
folder_out = 'out/'
fail_index = 'failed.csv'
#END Input


#FACTORS for haar cascades, tweaked by hand
#possibly quite image-specific, CERTAINLY cascade specific 
img_scale=2
min_size_face = (180,180)
max_size_face = (380,380)
haar_scale_face = 1.09
min_neighbors_face = 21
haar_flags_face = 0

min_size_eye = (100,100)
max_size_eye = (210,210)
haar_scale_eye = 1.0275
min_neighbors_eye = 32
haar_flags_eye = 0
#END FACTORS

globalpath = path.expanduser(globalpath)
img_path = globalpath + folder_in
out_path = globalpath +  folder_out
pictures = listdir(img_path)
lelist = []
lelist = list(lelist)

for i in pictures:
    picture, faceCasc, leyeCasc, reyeCasc = Load(img_path+i)
    
    gray = cv2.cvtColor(picture, cv2.cv.CV_BGR2GRAY)# Convert color input image to grayscale
    smallimg = cv2.resize(gray, (cv2.cv.Round(np.shape(picture)[1] / img_scale),cv2.cv.Round (np.shape(picture)[0] / img_scale)))# scale
    cv2.equalizeHist(smallimg, smallimg)# Equalize the histogram
     
    leye = leyeCasc.detectMultiScale(smallimg, haar_scale_eye, min_neighbors_eye, haar_flags_eye, min_size_eye, max_size_eye)
    reye = reyeCasc.detectMultiScale(smallimg, haar_scale_eye, min_neighbors_eye, haar_flags_eye, min_size_eye, max_size_eye)
    
    lpt=rpt=[]
    if np.shape(leye)[0] == 0:
        print 'No left eye position was retrieved from ' + img_path+i
        lelist += [i,'nle']
    elif np.shape(leye)[0] != 1:
        print 'Multiple left eye positions were retrieved from ' + img_path+i
        lelist += [i,'mle']
    else:
        x,y,w,h = tuple(leye[0])
        lpt = (int(x+w/2), int(y+h/2))
            
        if np.shape(reye)[0] == 0:
            print 'No right eye position was retrieved from ' + img_path+i
            lelist += [i,'nre']
        elif np.shape(reye)[0] != 1:
            print 'Multiple right eye positions were retrieved from ' + img_path+i
            lelist += [i,'mre']
        else:
            x,y,w,h = tuple(reye[0])
            rpt = (int(x+w/2), int(y+h/2))
    #            cv2.circle(picture, rpt, 10, cv2.cv.RGB(255, 0, 0), -1)
        
            if np.shape(lpt)[0] == 0 or np.shape(rpt)[0] == 0:
                pass
            else:
                cath = np.array(lpt) - np.array(rpt)
                ang = np.degrees(np.tan(cath[1]/cath[0]))
                picshape = picture.shape[:-1] #only x and y; NOT channel
                img_cent = tuple(np.array(picshape)/2)
                rot_mat = cv2.getRotationMatrix2D(img_cent,ang,1)
                picture = cv2.warpAffine(picture, rot_mat, picshape,flags=cv2.INTER_LINEAR) #"picture" rotated HERE
        
                gray = cv2.cvtColor(picture, cv2.cv.CV_BGR2GRAY) # Convert color input image to grayscale     
                smallimg = cv2.resize(gray, (cv2.cv.Round(np.shape(picture)[1] / img_scale),cv2.cv.Round (np.shape(picture)[0] / img_scale))) # scale 
                cv2.equalizeHist(smallimg, smallimg)# Equalize the histogram
                face = faceCasc.detectMultiScale(smallimg, haar_scale_face, min_neighbors_face, haar_flags_face, min_size_face, max_size_face)
                leye = leyeCasc.detectMultiScale(smallimg, haar_scale_eye, min_neighbors_eye, haar_flags_eye, min_size_eye, max_size_eye)
                reye = reyeCasc.detectMultiScale(smallimg, haar_scale_eye, min_neighbors_eye, haar_flags_eye, min_size_eye, max_size_eye)
                if np.shape(face)[0] == 0:
                    print 'No face position was retrieved from ' + img_path+i
                    lelist += [i,'nf']
                elif np.shape(face)[0] != 1:
                    print 'Multiple face positions were retrieved from ' + img_path+i
                    lelist += [i,'mf']
                else:
                    if np.shape(leye)[0] == 0:
                        print 'No post-rotation left eye position was retrieved from ' + img_path+i
                        lelist += [i,'nle2']
                    elif np.shape(leye)[0] != 1:
                        print 'Multiple post-rotation left eye positions were retrieved from ' + img_path+i
                        lelist += [i,'mle2']
                    else:
                        x,y,w,h = tuple(leye[0]*img_scale)
                        lpt = (int(x+w/2), int(y+h/2))
                        if np.shape(reye)[0] == 0:
                            print 'No post-rotation right eye position was retrieved from ' + img_path+i
                            lelist += [i,'nre2']
                        elif np.shape(reye)[0] != 1:
                            print 'Multiple post-rotation right eye positions were retrieved from ' + img_path+i
                            lelist += [i,'mre2']
                        else:
                            x,y,w,h = tuple(reye[0]*img_scale)
                            rpt = (int(x+w/2), int(y+h/2))
                            ct = (np.array(lpt) + np.array(rpt)) / 2
                            fr_face = tuple(face[0]*img_scale)
                            picture = picture[ct[1]-1.288*fr_face[3]:ct[1]+1.512*fr_face[3],ct[0]-1.05*fr_face[2]:ct[0]+1.05*fr_face[2]]
                            cv2.imwrite(out_path+i, picture)
lelist = np.reshape(lelist, (-1,2))
mk_ffile(globalpath, fail_index, lelist)          