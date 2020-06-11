# Python program to explain cv2.copyMakeBorder() method  
   
# importing cv2  
import cv2  
   
# path  
path = r'C:\Users\Rajnish\Desktop\geeksforgeeks\geeks.png'
   
# Reading an image in default mode 
image = cv2.imread('Flor.jpg') 
   
# Window name in which image is displayed 
window_name = 'Image'
  
# Using cv2.copyMakeBorder() method 
image = cv2.copyMakeBorder(image, 100, 100, 100, 100, cv2.BORDER_REPLICATE) 
  
# Displaying the image  
cv2.imshow(window_name, image) 

cv2.waitKey(0)
cv2.destroyAllWindows()