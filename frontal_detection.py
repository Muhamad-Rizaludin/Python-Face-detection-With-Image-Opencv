import cv2

#memuat gambar kedalam program
image = cv2.imread('group 2.jpg')
#cv2.imshow('Person', image)

#convert image kedalam gray scale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imshow('GrayScale Person',gray)

#menginisiasi classifier yang akan digunakan untuk mendeteksi kepala dan mulut dengan parameter file xml untuk deteksi mulut dan kepala
kepala = cv2.CascadeClassifier('haar_frontal_detect.xml')
mulut = cv2.CascadeClassifier('haar_mouth.xml')

deteksi_kepala= kepala.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5) #menggunkan varible untuk mendeteksi wajah

#untuk mengecek banyaknya kepala yang terdeteksi
total_kepala = 0
for (x,y,w,h) in deteksi_kepala:
    #menggambar persegi hijau untuk kepala yg terdeteksi
    cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), thickness=2)
    total_kepala = total_kepala + 1
    cv2.putText(image, ('%02dkepala Terdeteksi' % total_kepala), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0,0,255), 1, cv2.LINE_AA)

    #untuk mengecek banyaknya mulut yang terdeteksi dan mengambar persegi biru
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = image[y:y+h, x:x+w]
    deteksi_mulut = mulut.detectMultiScale(roi_gray)
    for (mx,my,mw,mh) in deteksi_mulut:
        cv2.rectangle(roi_color,(mx,my),(mx+mw,my+mh),(255,0,0),2)

print(total_kepala,' kepala terdeteksi')
cv2.imshow('kepala terdeteksi', image)

cv2.waitKey(0)
cv2.destroyAllWindows()