from scipy.misc import imread
from matplotlib import pyplot
from PIL import Image
from PIL import ImageFilter
from math import floor, sqrt
import  numpy as np

def smoothing(pic_pth,output_path,mask_size):
    image = Image.open(pic_pth)
    # create a mask of appropriate size for median filtering according to the size of the image
    M,N = image.size
    if mask_size == 0:
        mask_size = floor(5*sqrt((float(M)/768)*(float(N)/512)))

    mask = ImageFilter.MedianFilter(int(mask_size))
    smoothedImage = image.filter(mask)
    smoothedImage.save(output_path)

# Smoothing
# smoothing("./skin_data/melanoma/dermIS/LMM2_orig.jpg","out.jpg",0)
#
#
# grey = Image.open('out.jpg').convert('L').save('grey.jpg')
#


# Edge detection
im = Image.open('srm.png')
im.save('srm.jpg')
im = imread('srm.jpg')
# pix = image.load()
# print image.size

print im.shape
m,n,_ = im.shape
len = 20
background_mean = np.mean(im[0:len,0:len] + im[0:len,n-len:n] + im[m-len:m, 0:len])
print background_mean
#
# contour = Image.open("out.jpg").convert('L')
# print contour.size
contour = Image.new('L', (m,n))
pix = contour.load()

for i in range(0,contour.size[0]):
    for j in range(0,contour.size[1]):
        # print (np.mean(im[i,j]) - background_mean)**2
        if (np.mean(im[i,j]) - background_mean)**2 < 7300:
            pix[i,j] = 0
        else:
            pix[i,j] = 255

contour.show()
contour.save('contour.jpg')

# smoothing("contour.jpg","smoothed.jpg",0)
# pyplot.imshow(res)
# pyplot.show()

