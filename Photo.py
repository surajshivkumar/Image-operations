import cv2 as cv
import numpy as np 
from matplotlib import pyplot as plt

print('Welcome\n')
location = input('Enter the path of the source image\n')#Enter the full path
img = cv.imread(location,1)#loading the image in color
#img.shape = (height,width,channels)
height=img.shape[0]
width=img.shape[1]

print('Available Operations:\n')
print('1)Grayscale\n2)Threshold\n3)Smoothing\n4)Gradient Filtering(Laplacian)\n5)Edge Detection\n6)View Histogram\n7)Segmentation\n8)Blue\n9)Green\n10)Red\n11)Transposing\n12)Face Detection\n13)Adjust Brightness and Contrast\n')

while True:

	choice = input('Enter your choice\n')
	cv.imshow('Original image',img)

	if choice=='1':
		t_img = cv.imread(location,0)
		cv.imshow('Transformed Image',t_img)
	elif choice=='2':
		img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
		t_img = cv.adaptiveThreshold(img_gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,11,2)
		cv.imshow("Transformed Image",t_img)
	elif choice=='3':
		t_img = cv.GaussianBlur(img,(5,5),2)
		cv.imshow("Transformed Image",t_img)
	elif choice=='4':
		img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
		t_img =cv.Laplacian(img_gray,cv.CV_64F)
		cv.imshow("Transformed Image",t_img)
	elif choice=='5':
		#Edge detection using canny edge detector
		img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
		t_img =cv.Canny(img_gray,100,200)
		cv.imshow("Transformed Image",t_img)
	elif choice=='6':
		#Image histogram using matplotlib
		plt.hist(img.ravel(),256,[0,256]); plt.show()
	elif choice=='7':
		img_gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
		ret, thresh = cv.threshold(img_gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
		cv.imshow("Transformed Image",thresh)
	elif choice=='8':
		#Blue mask
		B,G,R=cv.split(img)
		zeros = np.zeros((height,width),dtype="uint8")
		cv.imshow("Blue",cv.merge([B,zeros,zeros]))
		t_img = cv.merge([B,zeros,zeros])
	elif choice=='9':
		#Green mask
		B,G,R=cv.split(img)
		zeros = np.zeros((height,width),dtype="uint8")
		cv.imshow("Blue",cv.merge([zeros,G,zeros]))
		t_img = cv.merge([zeros,G,zeros])
	elif choice=='10':
		#Red mask
		B,G,R=cv.split(img)
		zeros = np.zeros((height,width),dtype="uint8")
		cv.imshow("Blue",cv.merge([zeros,zeros,R]))
		t_img = cv.merge([zeros,zeros,R])
	elif choice=='11':
		t_img = cv.transpose(img)
		cv.imshow("Transformed Image",t_img)
	elif choice=='12':
		face_cascade=cv.CascadeClassifier("haarcascade_frontalface_alt.xml")
		gray_img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
		faces=face_cascade.detectMultiScale(gray_img,scaleFactor=1.06,minNeighbors=6)
		for x,y,w,h in faces:
			img1=cv.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)
			cv2.imshow("Gray",img1)
	elif choice=='13':
		cv.destroyAllWindows()
		new_image = np.zeros(img.shape, img.dtype)
		#g(x) = alpha*f(x) + beta 
		alpha = float(input('Enter alpha value [1.0-3.0]:\n'))
		beta = 	int(input('Enter beta value [0-100]:\n'))
		print('This may take a while....')
		for y in range(img.shape[0]):
			for x in range(img.shape[1]):
				for c in range(img.shape[2]):
					new_image[y,x,c] = np.clip(alpha*img[y,x,c] + beta, 0, 255)
		cv.imshow('Transformed Image',new_image)

	cv.waitKey(0)
	cv.destroyAllWindows()
	
	save = input('Would you like to save your image?(y/n)\n')#User can optionally save his image
	
	if save=='y':
		dest_path = input('Enter the destination path')
		cv.imwrite(dest_path+'.jpg',t_img)
	elif save=='n':
		pass
	else:
		print('invalid choice')
	exit_choice = input('Do you want to exit?(y/n)\n')
	if exit_choice=='y':
		exit(0)
	elif exit_choice=='n':
		print('')
	else:
		print('invalid input\n')