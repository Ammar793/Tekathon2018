import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import time

files = os.listdir(".//train/train")


for i in range(2551,len(files)):
	imag ='./train/train/'+files[i]
	
	img = cv2.imread(imag)
	
	img1= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	img2 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	
	#print(img2)
	h,w = img1.shape
	
	#print(h)
	#print(w)
	
#	keep_going = True
#	c =0
#	while(keep_going):
	
	
#		img1= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

	c2=0
	do_left=True
	while(do_left):	
		left = np.zeros((h,5*(c2+1),3),dtype=np.uint8)
		left2 = np.zeros((h,5*(c2+1),3),dtype=np.uint8)
		img_remain = np.zeros((h,w-5*(c2+1),3),dtype=np.uint8)
		img_remain2 = np.zeros((h,w-5*(c2+1),3),dtype=np.uint8)
		
		h_l,w_l,c_l = left.shape
		h_i,w_i,c_i = img_remain.shape
		
		for j in range (0,len(left)):
			for k in range (0,len(left[0])):
				left[j,k] = img[j,k]
				left2[j,k] = img2[j,k]
				
		for j in range (0,len(img_remain)):
			for k in range (0,len(img_remain[0])):
				img_remain[j,k]=img[j,k+5*(c2+1)]
				img_remain2[j,k]=img2[j,k+5*(c2+1)]
				
		edges_left=cv2.Canny(left,100,200)
		edges_remain=cv2.Canny(img_remain,100,200)
		
		ed_left=0
		ed_remain=0
		for j in range (0,len(left)):
			for k in range (0,len(left[0])):
			
				if(edges_left[j,k]==255):
					ed_left+=1
						
		for j in range (0,len(img_remain)):
			for k in range (0,len(img_remain[0])):
				if(edges_remain[j,k]==255):
					ed_remain+=1
		
		average_intensity_left = np.average(np.average(left2, axis=0),axis=0)		
		average_intensity_remain = np.average(np.average(img_remain2, axis=0),axis=0)		
		
		ed_left = np.round(ed_left/(h_l*w_l),2)
		ed_remain = np.round(ed_remain/(h_i*w_i),2)
		
		average_intensity_left = np.round(average_intensity_left,2)	
		average_intensity_remain = np.round(average_intensity_remain,2)	
		
		#print(ed_left)
		#print(ed_remain)
		##print(average_intensity_left)
		##print(average_intensity_remain)			
		r = abs(int(average_intensity_left[0])- int(average_intensity_remain[0]))
		g = abs(int(average_intensity_left[1])- int(average_intensity_remain[1]))
		b = abs(int(average_intensity_left[2])- int(average_intensity_remain[2]))
		
		#print(r)
		#print(g)
		#print(b)
		
		ed_diff = abs(ed_left-ed_remain)
		
		#print(ed_diff)
		if( r<1 or g<1 or b<1 or ed_diff<0.02 or c2>20):					
			do_left=False
			#print("done")
			
		
		
		c2+=1
			
	l = 5*c2

	
	c2=0
	do_right=True
	while(do_right):
		
		right = np.zeros((h,5*(c2+1),3),dtype=np.uint8)
		img_remain = np.zeros((h,w-5*(c2+1),3),dtype=np.uint8)		
		right2 = np.zeros((h,5*(c2+1),3),dtype=np.uint8)
		img_remain2 = np.zeros((h,w-5*(c2+1),3),dtype=np.uint8)
		
		h_l,w_l,c = right.shape
		h_i,w_i,c = img_remain.shape
		
		
		for j in range (0,len(right)):
			for k in range (0,len(right[0])):
				right[j,k] = img[j,(w-5*(c2+1))+k]
				right2[j,k] = img2[j,(w-5*(c2+1))+k]
				
		for j in range (0,len(img_remain)):
			for k in range (0,len(img_remain[0])):
				img_remain[j,k]=img[j,k]
				img_remain2[j,k]=img2[j,k]
				
		edges_right=cv2.Canny(right,100,200)
		edges_remain=cv2.Canny(img_remain,100,200)
		
		ed_right=0
		ed_remain=0
		for j in range (0,len(right)):
			for k in range (0,len(right[0])):
			
				if(edges_right[j,k]==255):
					ed_right+=1
						
		for j in range (0,len(img_remain)):
			for k in range (0,len(img_remain[0])):
				if(edges_remain[j,k]==255):
					ed_remain+=1
		
		average_intensity_right = np.average(np.average(right2, axis=0),axis=0)		
		average_intensity_remain = np.average(np.average(img_remain2, axis=0),axis=0)
		
		

		
		ed_right = np.round(ed_right/(h_l*w_l),2)
		ed_remain = np.round(ed_remain/(h_i*w_i),2)	
		#print(ed_right)
		#print(ed_remain)		
		
		ed_diff = abs(ed_right-ed_remain)
		r = abs(int(average_intensity_right[0])- int(average_intensity_remain[0]))
		g = abs(int(average_intensity_right[1])- int(average_intensity_remain[1]))
		b = abs(int(average_intensity_right[2])- int(average_intensity_remain[2]))
		
		#print(r)
		#print(g)
		#print(b)
		ed_diff = abs(ed_right-ed_remain)
		#print(ed_diff)
		if( r<1 or g<1 or b<1 or ed_diff<0.02 or c2>20 or c2*5 > (l-15)):						
			do_right=False
		
		c2+=1
			
	r = 5*c2

	c2=0
	do_top=True
	while(do_top):
		
		top = np.zeros((5*(c2+1),w,3),dtype=np.uint8)
		top2 = np.zeros((5*(c2+1),w,3),dtype=np.uint8)
		img_remain = np.zeros((h-5*(c2+1),w,3),dtype=np.uint8)
		img_remain2 = np.zeros((h-5*(c2+1),w,3),dtype=np.uint8)
		h_l,w_l,c = top.shape
		h_i,w_i,c = img_remain.shape
		for j in range (0,len(top)):
			for k in range (0,len(top[0])):
				top[j,k] = img[j,k]	
				top2[j,k] = img2[j,k]	
				
		for j in range (0,len(img_remain)):
			for k in range (0,len(img_remain[0])):
				img_remain[j,k]=img[j+5*(c2+1),k]
				img_remain2[j,k]=img2[j+5*(c2+1),k]
				
		edges_top=cv2.Canny(top,100,200)
		edges_remain=cv2.Canny(img_remain,100,200)
		
		ed_top=0
		ed_remain=0
		for j in range (0,len(top)):
			for k in range (0,len(top[0])):			
				if(edges_top[j,k]==255):
					ed_top+=1
						
		for j in range (0,len(img_remain)):
			for k in range (0,len(img_remain[0])):
				if(edges_remain[j,k]==255):
					ed_remain+=1
		
		average_intensity_top = np.average(np.average(top2, axis=0),axis=0)		
		average_intensity_remain = np.average(np.average(img_remain2, axis=0),axis=0)
		
		ed_top = np.round(ed_top/(h_l*w_l),2)
		ed_remain = np.round(ed_remain/(h_i*w_i),2)
		
		#print(ed_top)
		#print(ed_remain)
		c2+=1
		r = abs(int(average_intensity_top[0])- int(average_intensity_remain[0]))
		g = abs(int(average_intensity_top[1])- int(average_intensity_remain[1]))
		b = abs(int(average_intensity_top[2])- int(average_intensity_remain[2]))
		
		#print(r)
		#print(g)
		#print(b)
		ed_diff = abs(ed_top-ed_remain)
		#print(ed_diff)
		if( r<1 or g<1 or b<1 or ed_diff<0.02 or c2>20):			
			
			do_top=False
		
		
			
	t = 5*c2

	c2=0
	do_bottom=True
	while(do_bottom):
		
		bottom = np.zeros((5*(c2+1),w,3),dtype=np.uint8)
		img_remain = np.zeros((h-5*(c2+1),w,3),dtype=np.uint8)		
		bottom2 = np.zeros((5*(c2+1),w,3),dtype=np.uint8)
		img_remain2 = np.zeros((h-5*(c2+1),w,3),dtype=np.uint8)
		h_l,w_l,c = bottom.shape
		h_i,w_i,c = img_remain.shape
		
		for j in range (0,len(bottom)):
			for k in range (0,len(bottom[0])):
				bottom[j,k] = img[(h-5*(c2+1))+j,k]	
				bottom2[j,k] = img2[(h-5*(c2+1))+j,k]	
				
		for j in range (0,len(img_remain)):
			for k in range (0,len(img_remain[0])):
				img_remain[j,k]=img[j,k]
				img_remain2[j,k]=img2[j,k]
				
		edges_bottom=cv2.Canny(bottom,100,200)
		edges_remain=cv2.Canny(img_remain,100,200)
		
		ed_bottom=0
		ed_remain=0
		for j in range (0,len(bottom)):
			for k in range (0,len(bottom[0])):
			
				if(edges_bottom[j,k]==255):
					ed_bottom+=1
						
		for j in range (0,len(img_remain)):
			for k in range (0,len(img_remain[0])):
				if(edges_remain[j,k]==255):
					ed_remain+=1
		
		average_intensity_bottom = np.average(np.average(bottom, axis=0),axis=0)		
		average_intensity_remain = np.average(np.average(img_remain, axis=0),axis=0)
		
		
		ed_bottom = np.round(ed_bottom/(h_l*w_l),2)
		ed_remain = np.round(ed_remain/(h_i*w_i),2)

		#print(ed_bottom)
		#print(ed_remain)
		c2+=1
		
		r = abs(int(average_intensity_bottom[0])- int(average_intensity_remain[0]))
		g = abs(int(average_intensity_bottom[1])- int(average_intensity_remain[1]))
		b = abs(int(average_intensity_bottom[2])- int(average_intensity_remain[2]))
		
		#print(r)
		#print(g)
		#print(b)
		
		ed_diff = abs(ed_bottom-ed_remain)
		#print(ed_diff)
		if( r<1 or g<1 or b<1 or ed_diff<0.02 or c2>20 or c2*5 > (t -15)):			
			#print("done")			
			do_bottom=False
		
		
	print(t)
	
	b = 5*c2		
	print(b)				
					
	#print(l)
	#print(t)	
	#print(r)	
	#print(b)
	
	##print(w-r)
	##print(h-b)			
	img_square= cv2.rectangle(img,(l,t),(w-r,h-b),(0,255,0),1)
	cv2.imshow("square", img_square)
	cv2.waitKey(0)
	
	
	
	
	mask = np.zeros(img.shape[:2],np.uint8)
	bgdModel = np.zeros((1,65),np.float64)
	fgdModel = np.zeros((1,65),np.float64)
	#print((w-r)-l)
	#print((h-b)-t)
	
	
	
	rect = (l,t,(w-r)-l,(h-b)-t)
	
	for j in range (0,h):
		for k in range(0,w):
			if(j<t or j>(h-b)):
				mask[j,k]=0
			elif(k<l or k>(w-r)):
				mask[j,k]=0
			else:
				mask[j,k]=3
		

	print(rect)
	if(rect[2] >0 and rect[3]>0):
		cv2.grabCut(img,mask,rect,bgdModel,fgdModel,10,cv2.GC_INIT_WITH_RECT)
	
		mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
	
		img = img*mask2[:,:,np.newaxis]
	
		cv2.imwrite( ("segmented/"+files[i]), img );				
	
	#cv2.imwrite( ("segmented/"+files[i]), img );				
				
			
			
def blekh():			
	right = np.zeros((h,5*(c+1)),dtype=np.uint8)
	top = np.zeros((5*(c+1),w),dtype=np.uint8)
	bottom = np.zeros((5*(c+1),w),dtype=np.uint8)
	img_remain = np.zeros((h-5*(c+1),w-5*(c+1)),dtype=np.uint8)

	for j in range (0,len(left)):
		for k in range (0,len(left[0])):
			left[j,k] = img1[j,k]				
			right[j,k] = img1[j,(w-5*(c+1))+k]


	for j in range (0,len(top)):
		for k in range (0,len(top[0])):
			top[j,k] = img1[j,k]				
			bottom[j,k] = img1[(h-5*(c+1))+j,k]
			
			
	for j in range (0,len(img_remain)):
		for k in range (0,len(img_remain[0])):
			img_remain[j,k]=img1[j+5*(c+1),k+5*(c+1)]
			
			
	#cv2.imshow("b",bottom)
	#cv2.imshow("t",top)
	#cv2.imshow("l",left)
	#cv2.imshow("r",img_remain)

	#get edges
	edges_left=cv2.Canny(left,100,200)
	edges_right=cv2.Canny(right,100,200)
	edges_bottom=cv2.Canny(bottom,100,200)
	edges_top=cv2.Canny(top,100,200)
	edges=cv2.Canny(img_remain,100,200)


	#get edge density
	ed_left=0
	ed_right=0
	ed_top=0
	ed_bottom=0
	ed_remain=0
	for j in range (0,len(left)):
		for k in range (0,len(left[0])):
		
			if(edges_left[j,k]==255):
				ed_left+=1
				
			if(edges_right[j,k]==255):
				ed_right+=1
			

	for j in range (0,len(top)):
		for k in range (0,len(top[0])):
			if(edges_top[j,k]==255):
				ed_left+=1
				
			if(edges_bottom[j,k]==255):
				ed_right+=1
				

			
	for j in range (0,len(img_remain)):
		for k in range (0,len(img_remain[0])):
			if(edges[j,k]==255):
				ed_remain+=1
				
	#print(ed_left)
	#print(ed_remain)
	#print("break")

	#get avg intensity
	average_intensity_left = np.average(np.average(left, axis=0),axis=0)
	average_intensity_right = np.average(np.average(right, axis=0),axis=0)
	average_intensity_top = np.average(np.average(top, axis=0),axis=0)
	average_intensity_bottom = np.average(np.average(bottom, axis=0),axis=0)		
	average_intensity = np.average(np.average(img_remain, axis=0),axis=0)


	#print(average_intensity_bottom)
	#print(average_intensity)



	#		cv2.imshow("left_edges",edges_left)
	#		cv2.imshow("right_edges",edges_right)
	#		cv2.imshow("t_edges",edges_bottom)
	#		cv2.imshow("b_edges",edges_top)
	#		cv2.imshow("edges",edges)




	

	#		cv2.imshow("square", img_square)
	cv2.waitKey(0)
	c+=1



	if(c==10):
		keep_going=False