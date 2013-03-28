import cv2

def Load(picture):
    pic = cv2.imread(picture)
    faceCasc = cv2.CascadeClassifier('/usr/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
    leyeCasc = cv2.CascadeClassifier('/usr/share/OpenCV/haarcascades/haarcascade_lefteye_2splits.xml')
    reyeCasc = cv2.CascadeClassifier('/usr/share/OpenCV/haarcascades/haarcascade_righteye_2splits.xml')
    eyesCasc = cv2.CascadeClassifier('/usr/share/OpenCV/haarcascades/haarcascade_mcs_eyepair_small.xml')
    return(pic, faceCasc, leyeCasc, reyeCasc, eyesCasc)

def Display(image):
    cv2.namedWindow('Red Eye Test', flags=0)
    cv2.resizeWindow('Red Eye Test', 500,600)
    cv2.imshow('Red Eye Test', image)
    cv2.waitKey(0)
    cv2.destroyWindow('Red Eye Test')
    
def mk_ffile(globalpath, filename, fails):
    from csv import writer
    from os import path, makedirs
    from shutil import move
    from datetime import date, datetime
    jzt=datetime.now()
    time = str(date.today())+str(jzt.hour)+str(jzt.minute)+str(jzt.second)
    lefile=globalpath+filename
    if path.isfile(lefile):
        if path.isdir(globalpath+'.backup'):
            pass
        else: makedirs(globalpath+'.backup')        
        move(lefile, globalpath+'.backup/'+time+filename)
        print 'Moved  pre-existing fail file ('+lefile+') to backup location.'
    else: pass
    failfile = open(lefile, 'a')
    failwriter = writer(failfile, delimiter=',')
    for i in fails:
        failwriter.writerow(i)
    failfile.close()
    
def draw_featureareas(img_scale, picture, face, eyes):
    a,b,c,d = tuple(face[0]*img_scale)
    pt3 = (a, b)
    pt4 = (a+c, b+d)
    cv2.rectangle(picture, pt3, pt4, cv2.cv.RGB(0, 255, 0), 3, 8, 0)
    x,y,w,h = tuple(eyes[0]*img_scale)
    pt1 = (x, y)
    pt2 = (x+w, y+h)
    cv2.rectangle(picture, pt1, pt2, cv2.cv.RGB(255, 0, 0), 3, 8, 0)    

def lecenter(area):
    x,y,w,h = tuple(area[0])
    ct = (int(x+round(w/2)), int(y+round(h/2)))
    return ct
    
def lescale(values, upscale, downscale=1):
	from numpy import array
	values = array(values)
	values = values * upscale / downscale
	return values.astype(int)
