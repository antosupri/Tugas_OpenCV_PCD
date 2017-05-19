import numpy as np 			#untuk memanggil library numpy
import cv2 				#untuk memanggil library opencv
import matplotlib.pyplot as plt		#untuk memanggil library matplotlib
from scipy import ndimage 		#untuk memangil library ndimage dari scipy


img = cv2.imread('cat.jpg')			#untuk menmanggil gambar cat.jpg, yang akan ditampilkan
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)	#untuk mengkonvert gambar menjadi graysclae
data = np.array(gray, dtype=float)

kernel = np.array([[-1, -1, -1, -1, -1],
					[-1,  1,  2,  1, -1],
					[-1,  2,  4,  2, -1],
					[-1,  1,  2,  1, -1],
					[-1, -1, -1, -1, -1]])
highpass_5x5 = ndimage.convolve(data, kernel)					

hist1,bins1 = np.histogram(highpass_5x5.flatten(),256,[0,256]) 	
cdf1 = hist1.cumsum() 						#membuat histogram dari hasil gambar yang telah di high pass filter pada lpf
norm1 = cdf1 * hist1.max()/ cdf1.max()

cv2.imshow('Grayscale',gray)					#menampilkan gambar grayscale yang telah dikonvert
cv2.imshow('Highpass_5x5',highpass_5x5)				#menampilkan gambar hasil dari highpass filter yang telah dilakukan pada gambar
cv2.imshow('Org',img)						#menampilkan gambar asli sebelum dirubah dan di filter
plt.plot(norm1, color = 'b')					#menampilkan keterangan norm1 pada histogram
plt.hist(highpass_5x5.flatten(),256,[0,256], color = 'r')	#memplot dan menampilkan histogram
plt.xlim([0,256])
plt.legend(('cdf','histogram'), loc = 'upper left')		#memberi keterangan pada tampilan histogram

plt.show()							#untuk menampilkan histogram hasil high pass filter beserta keteranganya dalam sebuah frame

#ketika semua selesai maka gambar akan di close
cv2.waitKey(0)
cv2.destroyAllWindows()
