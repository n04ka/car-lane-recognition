from math import isnan
from os import name
import numpy as np
import cv2 as cv
import time

draw_lines = True
fps = 200
buffer_size = 25

k1s = []
k2s = []
b1s = []
b2s = []
vid = cv.VideoCapture('roadvideo1.mp4')
  

def cut(img, x0, x1, y0, y1): # x0, x1, y0, y1 are ratios
    height, width = img.shape[0], img.shape[1]
    return img[int(height*y0):int(height*y1), int(width*x0):int(width*x1)], (int(width*x0), int(height*y0))


def look_for_Hough(img, min_theta, max_theta, max_treshold=100, min_treshhold=20):
    for threshold in range(max_treshold, min_treshhold, -10):
        lines = cv.HoughLines(img, 1, np.pi/250, threshold, min_theta=min_theta, max_theta=max_theta)
        if lines is not None:
            return lines
    return None 


def main():
    while True:
        tic = time.perf_counter()
        ret, c_src  = vid.read()

        if not ret:
            print("No lines")
            continue
            
        src = cv.cvtColor(c_src, cv.COLOR_BGR2GRAY)

        area, bias = cut(src, 0.2, 0.8, 0.35, 0.8)
        blurred = cv.blur(area, (3, 3))
        bounds = cv.Canny(blurred, 50, 180, None, 3)

        lines1 = look_for_Hough(bounds, min_theta=np.pi/6, max_theta=np.pi/2*0.8)
        lines2 = look_for_Hough(bounds, min_theta=np.pi/2*1.2, max_theta=5*np.pi/6)
        
        if lines1 is None or lines2 is None:
            continue
            
        right = lines1[np.argmin(lines1[:, 0, 1])]
        left = lines2[np.argmax(lines2[:, 0, 1])]
        
        #print(f"left: (rho={left[0][0]}\ttheta={left[0][1]})\tright: (rho={right[0][0]}\ttheta={right[0][1]})")
        
        rho1, theta1= right[0]
        rho2, theta2 = left[0]
        k1 = -1/np.tan(theta1)
        b1 = rho1/np.sin(theta1)
        k2 = -1/np.tan(theta2)
        b2 = rho2/np.sin(theta2)
        
        if len(k1s) >= buffer_size:
            k1s.pop(0)
            b1s.pop(0)
            k2s.pop(0)
            b2s.pop(0)

        k1s.append(k1)
        k2s.append(k2)        
        b1s.append(b1)
        b2s.append(b2)
        
        k1 = np.mean(k1s)
        b1 = np.mean(b1s)
        k2 = np.mean(k2s)
        b2 = np.mean(b2s)
        
        #print(k2, b1, k1, b1, sep="\t")

        x = (b1-b2) / (k2-k1)   # intersection
        if isnan(x):
            x = 10000
        else:
            x = int(x)
        pt1 = (x + bias[0], int(k1*x + b1) + bias[1])

        x = int((area.shape[0]-b1)/k1) # right border
        pt2 = (x + bias[0], area.shape[0] + bias[1])

        x = int((area.shape[0]-b2)/k2) # left border
        pt3 = (x + bias[0], area.shape[0] + bias[1])

        x = np.mean((pt2[0], pt3[0]))
        mid_point = (int(x), pt2[1])

        div = -np.arctan((x-pt1[0])/(mid_point[1]-pt1[1]))
        cv.putText(c_src, "%.2f" % div, mid_point, cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))

        cv.line(c_src, pt1, pt2, (0, 0, 255), 5, cv.LINE_AA)
        cv.line(c_src, pt1, pt3, (0, 0, 255), 5, cv.LINE_AA)
        cv.line(c_src, pt1, mid_point, (0, 255, 0), 5, cv.LINE_AA)

        while time.perf_counter() - tic < 1/fps:
            pass
        cv.putText(c_src, str(int(1/(time.perf_counter() - tic))), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255))
        cv.imshow('Video', c_src)
        cv.imshow('Bounds', bounds)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv.destroyAllWindows()
    

if __name__ == '__main__':
    main()