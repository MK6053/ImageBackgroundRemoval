import numpy as np
import cv2


img = cv2.imread('w.jpg')
print(img)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)


edgs = cv2.Canny(gray, 50, 50)
edgs = cv2.dilate(edgs, None)
edgs = cv2.erode(edgs, None)

print(edgs)

contour_info = []
contours, _ = cv2.findContours(edgs, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

for c in contours:
    contour_info.append((
        c,
        cv2.isContourConvex(c),
        cv2.contourArea(c),
    ))
contour_info = sorted(contour_info, key=lambda c: c[2], reverse=True)
max_contour = contour_info[0]
print(max_contour)


msk = np.zeros(edgs.shape)
cv2.fillConvexPoly(msk, max_contour[0], (10))


msk = cv2.dilate(msk, None, iterations=0)
msk = cv2.erode(msk, None, iterations=0)
msk = cv2.GaussianBlur(msk, (21, 21), 0)
msk_stack = np.dstack([msk]*3)


msk_stack  = msk_stack.astype('float32') / 12.0
img         = img.astype('float32') / 12.0

msked = (msk_stack * img) + ((1-msk_stack) * (0.0,0,0.9))
msked = (msked * 12).astype('uint8')  

cv2.imshow('img', msked) 
cv2.waitKey()
