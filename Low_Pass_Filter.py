import numpy as np                                          #untuk memanggil library numpy
import cv2                                                  #untuk memanggil library opencv
from matplotlib import pyplot as plt                        #untuk memanggil library pyplot

img = cv2.imread('cat.jpg')                                 #untuk menmanggil gambar cat.jpg, yang akan ditampilkan
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)                 #untuk mengkonvert gambar menjadi graysclae

kernel = np.ones((5,5),np.float32)/25                       #matrix 5x5 yang berisikan angka 1, lalu dibagi 25 untuk low pass filter


lpf = cv2.filter2D(gray,-1,kernel)                          #low pass filter gambar dengan menggunakan matrix pada variable kernel
hist1,bins1 = np.histogram(lpf.flatten(),256,[0,256])       
cdf1 = hist1.cumsum()                                       #membuat histogram dari hasil gambar yang telah di low pass filter pada lpf
norm1 = cdf1 * hist1.max()/ cdf1.max()                      
equ = cv2.equalizeHist(lpf)                                 #mengolah gambar dengan histogram equalization
res = np.hstack((lpf,equ))

cv2.imwrite('res.png',res)                                  #menyimpan gambar histogram equalization satu folder dengan lokasi program
cv2.imshow('Org',img)                                       #menampilkan gambar asli sebelum dirubah dan di filter
cv2.imshow('Gray',gray)                                     #menampilkan gambar grayscale yang telah dikonvert
cv2.imshow('LPF 5x5',lpf)                                   #menampilkan gambar yang sudah di low pass filter


plt.plot(norm1, color = 'b')                                #menampilkan keterangan norm1 pada histogram
plt.hist(lpf.flatten(),256,[0,256], color = 'r')            #memplot dan menampilkan histogram
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')         #memberi keterangan pada tampilan histogram
plt.show()                                                  #menampilkan histogram gambar hasil low pass filter  beserta keteranganya dalam sebuah frame


#ketika semua selesai maka gambar akan di close
cv2.waitKey(0)
cv2.destroyAllWindows()
