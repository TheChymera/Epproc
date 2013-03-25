#!/usr/bin/env python
from __future__ import division
__author__ = 'Horea Christian'
from os import path, listdir, walk
from collections import Counter 
import subprocess
from funct import Load, mk_ffile, lecenter
import cv2
import numpy as np

#INPUT
profile = 'processing/profiles/faces.pp3'# .pp3 file for rawtherapee RAW to PNG processing (as relative to global dir)
globalpath = '~/Data/face-pictures/'
folder_in = 'fin/'
folder_out = 'out/'
opencvpath = '.opencv/'
fail_index = 'failed.csv'
img_w =  0.87 # half of result image width (relative to detected face size) max 2 decimals!
img_h_up = 1 #result image height above middle of eyes (relative to detected face size) max 2 decimals!
img_h_down = 1.32 #result image height below middle of eyes (relative to detected face size) max 2 decimals!
#END Input


#FACTORS for haar cascades, tweaked by hand
#possibly image-specific, CERTAINLY cascade specific 
img_scale=2
min_size_face = (400,400)
max_size_face = (800,800)
haar_scale_face = 1.3
min_neighbors_face = 7
haar_flags_face = 0

min_size_eye = (120,120)
max_size_eye = (400,400)
haar_scale_eye = 1.25
min_neighbors_eye = 13
haar_flags_eye = 0

min_size_eyes = (120,120)
max_size_eyes = (650,650)
haar_scale_eyes = 1.01
min_neighbors_eyes = 2
haar_flags_eyes = 0
#END FACTORS

globalpath = path.expanduser(globalpath)
img_path = globalpath + folder_in
out_path = globalpath +  folder_out
profile = globalpath + profile
lelist = []
lelist = list(lelist)
w1 = img_w*100
h1 = img_h_up*100
h2 = img_h_down*100

for leroot, dirs, files in walk(img_path, topdown=False): #browse the folder which contains the imges (including subfolders)
    for directory in dirs:
        for i in listdir(leroot+'/'+directory):
            i=leroot+directory+'/'+i
            if path.splitext(i)[0][-1]+path.splitext(i)[1] not in ['q.NEF','i.NEF','o.NEF','p.NEF','a.NEF','s.NEF','d.NEF']:
                continue
            if Counter(path.splitext(x)[0] for x in listdir(leroot+'/'+directory))[path.splitext(i)[0]] == 1: # only run rawtherapee if the RAW files are 
                subprocess.call(['rawtherapee', '-o', leroot+directory, '-p', profile, '-n', '-Y', '-c', i])  # the only ones with their particular identifier
            i,_ = path.splitext(i) # name of in put png
            i = i + '.png'
            n,_ = path.splitext(i) # nam of output png
            n = n +'x.png'
            picture, faceCasc, leyeCasc, reyeCasc, eyesCasc = Load(i)
            
            gray = cv2.cvtColor(picture, cv2.cv.CV_BGR2GRAY)# Convert color input image to grayscale
            smallimg = cv2.resize(gray, (cv2.cv.Round(np.shape(picture)[1] / img_scale),cv2.cv.Round (np.shape(picture)[0] / img_scale)))# scale
            cv2.equalizeHist(smallimg, smallimg)# Equalize the histogram
             
            leye = leyeCasc.detectMultiScale(smallimg, haar_scale_eye, min_neighbors_eye, haar_flags_eye, min_size_eye, max_size_eye)
            reye = reyeCasc.detectMultiScale(smallimg, haar_scale_eye, min_neighbors_eye, haar_flags_eye, min_size_eye, max_size_eye)
            
            lpt=rpt=[]
            if np.shape(leye)[0] == 0:
                print 'No left eye position was retrieved from ' + i
                lelist += [i,'nle']
            elif np.shape(leye)[0] != 1:
                print 'Multiple left eye positions were retrieved from ' + i
                lelist += [i,'mle']
            else:
                x,y,w,h = tuple(leye[0])
                lpt = (int(x+w/2), int(y+h/2))
                    
                if np.shape(reye)[0] == 0:
                    print 'No right eye position was retrieved from ' + i
                    lelist += [i,'nre']
                elif np.shape(reye)[0] != 1:
                    print 'Multiple right eye positions were retrieved from ' + i
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
                        picshape = (picture.shape[1], picture.shape[0]) #only x and y; NOT channel. INVERTED so as to not rotate coordinates by 90deg
                        img_cent = tuple(np.array(picshape)/2)
                        rot_mat = cv2.getRotationMatrix2D(img_cent,ang,1)
                        picture = cv2.warpAffine(picture, rot_mat, picshape,flags=cv2.INTER_LINEAR) #"picture" rotated HERE
                
                        gray = cv2.cvtColor(picture, cv2.cv.CV_BGR2GRAY) # Convert color input image to grayscale     
                        smallimg = cv2.resize(gray, (cv2.cv.Round(np.shape(picture)[1] / img_scale),cv2.cv.Round (np.shape(picture)[0] / img_scale))) # scale 
                        cv2.equalizeHist(smallimg, smallimg)# Equalize the histogram
                        face = faceCasc.detectMultiScale(smallimg, haar_scale_face, min_neighbors_face, haar_flags_face, min_size_face, max_size_face)
                        eyes = eyesCasc.detectMultiScale(smallimg, haar_scale_eyes, min_neighbors_eyes, haar_flags_eyes, min_size_eyes, max_size_eyes)
                        if np.shape(face)[0] == 0:
                            print 'No face position was retrieved from ' + i
                            lelist += [i,'nf']
                        elif np.shape(face)[0] != 1:
                            print 'Multiple face positions were retrieved from ' + i
                            lelist += [i,'mf']
                        else:
                            if np.shape(eyes)[0] == 0:
                                print 'No post-rotation eyes position was retrieved from ' + i
                                lelist += [i,'nle2']
                            elif np.shape(eyes)[0] != 1:
                                print 'Multiple post-rotation eyes positions were retrieved from ' + i
                                lelist += [i,'mle2']
                            else:
                                ct = np.array(lecenter(eyes))*img_scale #center of eyes field
                                ctf = np.array(lecenter(face))*img_scale #center of face
                                fr_face = tuple(face[0] * img_scale)
                                div_w = round(fr_face[3]/100) # divided face width (to ensure no rounding errors or varying proportions)
                                a = ct[0]-w1*div_w
                                b = ct[0]+w1*div_w
                                c = ct[1]-h1*div_w
                                d = ct[1]+h2*div_w
                                picture = picture[c:d,a:b,:]
                                gray = cv2.cvtColor(picture, cv2.cv.CV_BGR2GRAY) # Convert color input image to grayscale
                                (im_bw, premsk) = cv2.threshold(gray, 205, 255, cv2.THRESH_BINARY) # make cut-off mask
                                kernel1 = (50,50)
                                kernel2 = (100,100)
                                premsk =  cv2.blur(premsk,kernel1) # blur mask (to enlarge radius)
                                (im_bw, premsk) = cv2.threshold(premsk, 252, 255, cv2.THRESH_BINARY_INV) # make cut-off from blur
                                circ_msk = np.zeros((np.shape(picture)[0], np.shape(picture)[1]), np.uint8)
                                ctf_circ = (ctf[0], ctf[1]-40*div_w) #center for the mask circle (masking everything below the neck)
                                ctf_circ = (int(ctf_circ[0]-(ct[0]-w1*div_w)), int(ctf_circ[1]-(ct[1]-h1*div_w))) #coordinates adjusted to post-crop pic
                                cv2.ellipse(circ_msk, ctf_circ, (int(73*div_w),int(122*div_w)),0,0,360,cv2.cv.Scalar(255,0,0), -1  )
                                msk = cv2.bitwise_and(premsk,premsk,mask = circ_msk)
                                msk = cv2.erode(msk,None, iterations=5)
                                msk = cv2.blur(msk,kernel2)
                                msk = msk/255
                                for z in np.arange(3): # apply "msk" as alpha-mask
                                    picture[:,:,z] = 255-(255-picture[:,:,z])*msk 
                                cv2.imwrite(n, picture)
lelist = np.reshape(lelist, (-1,2))
mk_ffile(globalpath+opencvpath, fail_index, lelist)          