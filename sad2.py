import cv2
import numpy as np

# Load the stereo images
img = cv2.imread('left.png')
img2 = cv2.imread('right.png')
comp = np.hstack((img,img2))
cv2.imwrite('comp.png',comp)
from PIL import Image
f=Image.open("comp.png")
f.show()

# convert stereo images to grayscale
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
# get the size of the images
# l -> lines
# c -> columns
# v -> channel (RGB)
l,c,v = img.shape

# initialize arrays
minSAD = np.ones((l,c)) * 999999
sad = np.ones((l,c))
sad1 = np.ones((l,c))
winsad = np.ones((l,c))
disp = np.zeros((l,c))
distance = np.zeros((l,c))

max_shift = 60

# set size of the SAD window
w_l = 7
w_c = 7

for shift in range(max_shift):
    print("New Shift: %d"%(shift))
    for u in range(0,l):
        for v in range(0,c):
            # calculate SAD
            #if(v+shift <= l):
 		if (int(w_c/2) <= v):
 			d = v - int(shift/2)
		else: 
			if (int(shift/4) <= v) :
				d = v - int(shift/4)
			else:
				d = v
		if (d + shift < c):
            sad[u,v] = np.abs((int(gray[u,v]) - int(gray2[u,d+shift])))
			distance [u,v] = np.abs(v - d+shift)


    for u in range (0,l):
        for v in range (0,c):
            sum_sad = 0
            	for d in range(w_l):
                    for e in range(w_c):
                   		 if(u+d < l and v+e < c):
                        		sum_sad +=  sad[u+d,v+e]

		#Save disparity
		if(sum_sad < minSAD[u,v]):
			minSAD[u,v] = winsad[u,v]
			disp[u,v] = distance[u,v]


print("Process Complete")
#write disparity map to image
cv2.imwrite('sad.png',disp)
print("Disparity Map Generated1")
img3 = cv2.imread('sad.png',0)
print("Disparity Map Generated2")
equ = cv2.equalizeHist(img3)
#equ=img3
print("Disparity Map Generated3")
cv2.imwrite('equ.png',equ)

g=Image.open("equ.png")
g.show()

