import numpy as np
import cv2

def resize(src,dstsize):
	if src.ndim==3:
		dstsize.append(3)
	dst=np.array(np.zeros(dstsize),src.dtype)
	factory=float(np.size(src,0))/dstsize[0] 
	factorx=float(np.size(src,1))/dstsize[1]
	print 'factory',factory,'factorx',factorx
	srcheight=np.size(src,0)
	srcwidth=np.size(src,1)
	print 'srcwidth',srcwidth,'srcheight',srcheight
	for i in range(dstsize[0]):
		for j in range(dstsize[1]):
			y=float(i)*factory
			x=float(j)*factorx
			if y+1>srcheight:  
				y-=1
			if x+1>srcwidth:
				x-=1 
			cy=np.ceil(y)
			fy=cy-1
			cx=np.ceil(x)
			fx=cx-1
			w1=(cx-x)*(cy-y)
			w2=(x-fx)*(cy-y)
			w3=(cx-x)*(y-fy)
			w4=(x-fx)*(y-fy)	
			if (x-np.floor(x)>1e-6 or y-np.floor(y)>1e-6): 
				t=src[fy,fx]*w1+src[fy,cx]*w2+src[cy,fx]*w3+src[cy,cx]*w4
				t=np.ubyte(np.floor(t))
				dst[i,j]=t

			else:   	
				dst[i,j]=(src[y,x])
	return dst
        

def test1():
    img=cv2.imread('cat.jpg',1)
    print np.size(img,0)
    dst=resize(img,[100,100])
    cv2.imshow('dst',dst)
    cv2.imshow('src',img)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__=='__main__':
    test1()
