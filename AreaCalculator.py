from PIL import Image

Image.MAX_IMAGE_PIXELS = None
image = Image.open("Image.png")
pixels_red_2 = pixels_red_1 = pixels_blue_1=pixels_blue_2 = pixels_green_1=pixels_green_2 = pixels_yellow_1 = pixels_yellow_2=pixels_purple_1= pixels_purple_2 = 0

"""
We color coded each sample with respect to its control square(of 1 cm^2)
The program counts each pixel that matches the colors used in the mould tracing
It then transforms the area from pixels into cm^2 using the control square
"""

for pixel in image.getdata():
    if pixel == (255, 0 , 0):
            pixels_red_1 += 1
    if pixel == (255, 100 , 0):
            pixels_red_2 += 1
    if pixel == (0, 255 , 0):
            pixels_green_1 += 1
    if pixel == (0, 255 , 100):
            pixels_green_2 += 1
    if pixel == (0, 0 , 255):
            pixels_blue_1 += 1
    if pixel == (100, 0 , 255):
            pixels_blue_2 += 1
    if pixel == (255, 255 , 0):
            pixels_yellow_1 += 1
    if pixel == (255, 255 , 100):
            pixels_yellow_2 += 1
    if pixel == (177, 156, 217):
        pixels_purple_1 += 1
    if pixel == (138, 85, 255):
        pixels_purple_2 += 1

area_1 = pixels_red_1/pixels_red_2 # area  of sample 1 in cm^2
area_2 = pixels_blue_1/pixels_blue_2
area_3 = pixels_green_1/pixels_green_2
area_4 = pixels_yellow_1/pixels_yellow_2
area_5 = pixels_purple_1/pixels_purple_2
print("Area 1 = " , area_1)
print("Area 2 = " , area_2)
print("Area 3 = " , area_3)
print("Area 4 = " , area_4)
print("Area 5 = " , area_5)
