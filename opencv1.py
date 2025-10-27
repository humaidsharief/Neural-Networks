import cv2
from matplotlib import pyplot as plt

#img = cv2.imread("other1.jpg")

#img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

#face_data = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_alt2.xml")
#faces = face_data.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5)
#print(faces)

#for (x, y, width, height) in faces:
#    cv2.circle(img_rgb, (x + (width//2), y + (height//2)), width//2,(0, 255, 0), 5)

#plt.subplot(1, 1, 1)
#plt.imshow(img_rgb)
#plt.show()


img = cv2.imread("Dataset/other/other2.jpg")

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

car_data = cv2.CascadeClassifier("haarcascades/haarcascade_cars.xml")
STOP_data = cv2.CascadeClassifier("haarcascades/haarcascade_stopsign.xml")

stop_coords = []
car_coords = []

try:
    car_coords = car_data.detectMultiScale(img_gray, minSize=(20,20)).tolist()
    print("Car found", car_coords)
except:
    print("No cars nearby")

try:
    stop_coords = STOP_data.detectMultiScale(img_gray, minSize=(20,20)).tolist()
    print("Stop sign found", stop_coords)
except:
    print("No stop signs found")

img_height, img_width, img_channels = img.shape
left_border = img_width/2
right_border = img_width

for (x, y, width, height) in stop_coords:
    cv2.circle(img_rgb, (x + (width//2), y + (height//2)), width//2,(0, 255, 0), 5)

def check_forward(car_coords, stop_coords):
    if len(stop_coords) != 0:
        return False
    elif len(car_coords) == 0:
        return True
    else:
        for (x, y, width, height) in car_coords:
            if x > left_border and x + width < right_border:
                if width / img_width > 0.15:
                    print("Car too close")
                    return False
        return True

plt.subplot(1, 1, 1)
plt.imshow(img_rgb)
plt.show()

print("Drive ahead?", check_forward(car_coords, stop_coords))