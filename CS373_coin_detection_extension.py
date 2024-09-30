# Built in packages
import math
import sys

# Matplotlib will need to be installed if it isn't already. This is the only package allowed for this base part of the 
# assignment.
from matplotlib import pyplot
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

# import our basic, light-weight png reader library
import imageIO.png

# Define constant and global variables
TEST_MODE = False    # Please, DO NOT change this variable!

# method to read a PNG file and return the image width, height and pixel arrays for RGB components
def readRGBImageToSeparatePixelArrays(input_filename):
    image_reader = imageIO.png.Reader(filename=input_filename)
    # png reader gives us width and height, as well as RGB data in image_rows (a list of rows of RGB triplets)
    (image_width, image_height, rgb_image_rows, rgb_image_info) = image_reader.read()

    print("read image width={}, height={}".format(image_width, image_height))

    # our pixel arrays are lists of lists, where each inner list stores one row of greyscale pixels
    pixel_array_r = []
    pixel_array_g = []
    pixel_array_b = []

    for row in rgb_image_rows:
        pixel_row_r = []
        pixel_row_g = []
        pixel_row_b = []
        r = 0
        g = 0
        b = 0
        for elem in range(len(row)):
            # RGB triplets are stored consecutively in image_rows
            if elem % 3 == 0:
                r = row[elem]
            elif elem % 3 == 1:
                g = row[elem]
            else:
                b = row[elem]
                pixel_row_r.append(r)
                pixel_row_g.append(g)
                pixel_row_b.append(b)

        pixel_array_r.append(pixel_row_r)
        pixel_array_g.append(pixel_row_g)
        pixel_array_b.append(pixel_row_b)

    return (image_width, image_height, pixel_array_r, pixel_array_g, pixel_array_b)

# method to create a list of lists based array representation for an image, initialized with a value
def createInitializedGreyscalePixelArray(image_width, image_height, initValue = 0):
    new_pixel_array = []
    for _ in range(image_height):
        new_row = []
        for _ in range(image_width):
            new_row.append(initValue)
        new_pixel_array.append(new_row)

    return new_pixel_array


###########################################
### You can add your own functions here ###
###########################################

# method to convert RGB pixel arrays to greyscale pixel array
def computeRGBToGreyscale(pixel_array_r, pixel_array_g, pixel_array_b, image_width, image_height):
    print("greyscaling image")
    greyscale_pixel_array = createInitializedGreyscalePixelArray(image_width, image_height)
    
    # convert the RGB pixel arrays to greyscale pixel array
    for i in range(0,image_height):
        for j in range(0,image_width):
            greyscale_pixel_array[i][j] = round(0.299*pixel_array_r[i][j]+ 0.587*pixel_array_g[i][j] + 0.114*pixel_array_b[i][j])

    return greyscale_pixel_array

# method to normalise the greyscale pixel array to a range of 0-255
def normaliseImage(image, image_width, image_height):
    print("normalising image")
    greyscale_pixel_array = image

    # get histogram frequency for q by flattening the greyscale pixel array
    q = []
    flattenedImage = [element for element2 in greyscale_pixel_array for element in element2]
    q = list(set(flattenedImage))

    # get histogram hq
    hq = []
    for b in (q):
        hq.append(flattenedImage.count(b))
        
    # get cumulative histogram cq
    cum = []
    for b in range(0,len(hq)):
        if b==0:
            cum.append(hq[0])
        else:
            cum.append(cum[b-1] + hq[b])

    numPixels = image_width*image_height;
    alpha = 0.05*numPixels
    beta = 0.95*numPixels

    #find cq and then qAlpha
    for i in range(0,len(cum)):
        if cum[i] > alpha and cum[i-1] < alpha:
            qAlpha = q[i+1]
            break

    #find cq and then qBeta
    for i in range(0,len(cum)):
        if cum[i] < beta and cum[i+1] >= beta:
            qBeta = q[i]
            break

    #calculate image value
    for i in range(0,image_height):
        for j in range(0,image_width):
            imageValue = (255/(qBeta-qAlpha))*(greyscale_pixel_array[i][j]-qAlpha)
            greyscale_pixel_array[i][j] = round(max(0, min(255, imageValue)))

    return greyscale_pixel_array

# EXTENSION
# method to compute histogram equalisation on the greyscale pixel array
def histogramEqualisationEXTENSION(image, image_width, image_height):
    t = image

    # get histogram frequency for q by flattening the greyscale pixel array
    q = []
    flattenedImage = [element for element2 in image for element in element2]
    q = list(set(flattenedImage))

    # get histogram hq
    hq = []
    for b in (q):
        hq.append(flattenedImage.count(b))
        
    # get cumulative histogram cq
    cum = []
    for b in range(0,len(hq)):
        if b==0:
            cum.append(hq[0])
        else:
            cum.append(cum[b-1] + hq[b])

    cMin = cum[0]
    cMax = cum[len(cum)-1]

    # generate the equalised histogram for new q
    qEqualised = []
    flattenedImage = [element for element2 in t for element in element2]
    qEqualised = list(set(flattenedImage))

    # get histogram hq
    hqEqualised = []
    for b in (qEqualised):
        hqEqualised.append(flattenedImage.count(b))

    # get cumulative histogram cq
    cumEqualised = []
    for b in range(0,len(hqEqualised)):
        if b==0:
            cumEqualised.append(hqEqualised[0])
        else:
            cumEqualised.append(cumEqualised[b-1] + hqEqualised[b])

    # calculate equalisation
    for i in range(0,image_height):
        for j in range(0,image_width):

            #get the index of the cum associated with the greyscale pixel value
            flattened_index = qEqualised.index(t[i][j])
            t[i][j] = round( 255*((cum[flattened_index] - cMin)/(cMax - cMin) ))

    # plt.bar(range(len(cum)), cum)
    # plt.xlabel('Bin')
    # plt.ylabel('Frequency')
    # plt.title('cumulative Histogram')
    # plt.show()

    # plt.bar(range(len(cumEqualised)), cumEqualised)
    # plt.xlabel('Bin')
    # plt.ylabel('Frequency')
    # plt.title('equalised cumulative Histogram')
    # plt.show()
    return t

# EXTENSION
# method to apply laplacian filter for edge detection on the greyscale pixel array
def laplacianFilterEXTENSION(image, image_width, image_height):
    print("applying laplace filter")
    laplacian = [[1.0, 1.0, 1.0],
                 [1.0, -8.0, 1.0],
                 [1.0, 1.0, 1.0]]

    # create a new image array to store the sharr filter result
    filtered = createInitializedGreyscalePixelArray(image_width, image_height)

    # apply the laplacian filter to the image
    for i in range(0, image_height):
        for j in range(0, image_width):
            if i-1 < 0 or i+1 >= image_height or j-1 < 0 or j+1 >= image_width:
                filtered[i][j] = 0
            else:
                filtered[i][j] = abs(round((image[i-1][j-1]*laplacian[0][0] + image[i-1][j]*laplacian[0][1] + image[i-1][j+1]*laplacian[0][2] + image[i][j-1]*laplacian[1][0] + image[i][j]*laplacian[1][1] + image[i][j+1]*laplacian[1][2] + image[i+1][j-1]*laplacian[2][0] + image[i+1][j]*laplacian[2][1] + image[i+1][j+1]*laplacian[2][2])))
    
    return filtered

# EXTENSION
# method to apply median filter for noise reduction on the greyscale pixel array
def medianFilterEXTENSION(image, image_width, image_height):
    print("applying median filter")
    padding = createInitializedGreyscalePixelArray(image_width+4, image_height+4)
    paddedErosion = createInitializedGreyscalePixelArray(image_width+4, image_height+4)
    depadded = createInitializedGreyscalePixelArray(image_width, image_height)
    medianList = []
    
    # adding borderZeroPadding
    for i in range(0, image_height):
        for j in range(0, image_width):
            padding[i+2][j+2] = image[i][j]
    
    #apply filter
    for i in range(2, image_height+2):
        for j in range(2, image_width+2):
            # apply 5x5 median filter
            medianList = [padding[i-2][j-2] , padding[i-2][j-1] , padding[i-2][j] , padding[i-2][j+1] , padding[i-2][j+2] , padding[i-1][j-2] , padding[i-1][j-1] , padding[i-1][j] , padding[i-1][j+1] , padding[i-1][j+2] , padding[i][j-2] , padding[i][j-1] , padding[i][j] , padding[i][j+1] , padding[i][j+2] , padding[i+1][j-2] , padding[i+1][j-1] , padding[i+1][j] , padding[i+1][j+1] , padding[i+1][j+2] , padding[i+2][j-2] , padding[i+2][j-1] , padding[i+2][j] , padding[i+2][j+1] , padding[i+2][j+2]]
            median = (sorted(medianList))[12]
            paddedErosion[i][j] = median

    #removing borderZeroPadding
    for i in range(0, image_height):
        for j in range(0, image_width):
            depadded[i][j] = paddedErosion[i+2][j+2]
     
    return depadded

# method to apply a 5x5 mean blur filter
def blurImage(image, image_width, image_height):
    print("blurring image")
    blurred = []
    
    #apply 5x5 mean blur filter ignoring borders    
    for i in range(0, image_height):
        temp = []
        for j in range(0, image_width):
            if i+2 >= image_height or j+2 >= image_width or j-2 < 0 or i-2 < 0:
                temp.append(0)
            else:
                sum = image[i-2][j-2] + image[i-2][j-1] + image[i-2][j] + image[i-2][j+1] + image[i-2][j+2] + image[i-1][j-2] + image[i-1][j-1] + image[i-1][j] + image[i-1][j+1] + image[i-1][j+2] + image[i][j-2] + image[i][j-1] + image[i][j] + image[i][j+1] + image[i][j+2] + image[i+1][j-2] + image[i+1][j-1] + image[i+1][j] + image[i+1][j+1] + image[i+1][j+2] + image[i+2][j-2] + image[i+2][j-1] + image[i+2][j] + image[i+2][j+1] + image[i+2][j+2]
                temp.append(abs(round(sum/25)))
        blurred.append(temp)

    return blurred

# EXTENSION
# method to apply a 5x5 gaussian blur filter
def gaussianBlurEXTENSION(image, image_width, image_height):
    print("applying gaussian blur")
    filtered = createInitializedGreyscalePixelArray(image_width, image_height)

    gaussian = [[1, 2, 4, 2, 1],
                [2, 4, 8, 4, 2],
                [4, 8, 16, 8, 4],
                [2, 4, 8, 4, 2],
                [1, 2, 4, 2, 1]]
    
    for i in range(0, image_height):
        for j in range(0, image_width):
            if i-2 < 0 or i+2 >= image_height or j-2 < 0 or j+2 >= image_width:
                filtered[i][j] = 0
            else:
                filtered[i][j] = abs(round((image[i-2][j-2]*gaussian[0][0] + image[i-2][j-1]*gaussian[0][1] + image[i-2][j]*gaussian[0][2] + image[i-2][j+1]*gaussian[0][3] + image[i-2][j+2]*gaussian[0][4] + image[i-1][j-2]*gaussian[1][0] + image[i-1][j-1]*gaussian[1][1] + image[i-1][j]*gaussian[1][2] + image[i-1][j+1]*gaussian[1][3] + image[i-1][j+2]*gaussian[1][4] + image[i][j-2]*gaussian[2][0] + image[i][j-1]*gaussian[2][1] + image[i][j]*gaussian[2][2] + image[i][j+1]*gaussian[2][3] + image[i][j+2]*gaussian[2][4] + image[i+1][j-2]*gaussian[3][0] + image[i+1][j-1]*gaussian[3][1] + image[i+1][j]*gaussian[3][2] + image[i+1][j+1]*gaussian[3][3] + image[i+1][j+2]*gaussian[3][4] + image[i+2][j-2]*gaussian[4][0] + image[i+2][j-1]*gaussian[4][1] + image[i+2][j]*gaussian[4][2] + image[i+2][j+1]*gaussian[4][3] + image[i+2][j+2]*gaussian[4][4])/100))
    
    return filtered

# EXTENSION
# method to apply adaptive thresholding algorithm to convert the image to a binary image
def adaptiveThresholdImageEXTENSION(image, image_width, image_height):
    print("calculating adaptive threshold")
    # get histogram frequency for q by flattening the greyscale pixel array
    q = []
    flattenedImage = [element for element2 in image for element in element2]
    q = list(set(flattenedImage))

    # get histogram hq
    hq = []
    for b in (q):
        hq.append(flattenedImage.count(b))
        
    # get cumulative histogram cq
    cum = []
    for b in range(0,len(hq)):
        if b==0:
            cum.append(hq[0])
        else:
            cum.append(cum[b-1] + hq[b])

    # calculate qhq
    qHq = []
    for i in range(0,len(q)):
        qHq.append(q[i]*hq[i])

    threshold1 = sum(qHq)/sum(hq)
    thresholds = []
    loop = True

    # loop until the threshold found is the same as the last threshold
    while loop == True:
        if thresholds.count(threshold1) > 1 and thresholds[len(thresholds)-1]==thresholds[len(thresholds)-2]:
            loop = False
            break

        hq_ob = []
        qhq_ob = []
        hq_bg = []
        qhq_bg = []

        # gathering values for object and background
        for i in range(0,len(hq)):
            if q[i] > threshold1:
                hq_ob.append(hq[i])
                qhq_ob.append(q[i]*hq[i])
            if q[i] <= threshold1:
                hq_bg.append(hq[i])
                qhq_bg.append(q[i]*hq[i])

        # calculate average intensity for object and background
        avg_ob = sum(qhq_ob)/sum(hq_ob)
        avg_bg = sum(qhq_bg)/sum(hq_bg)
        threshold2 = round((avg_ob + avg_bg)/2)
        threshold1 = threshold2
        thresholds.append(threshold2)

    # apply threshold to the image
    for i in range(0, image_height):
        for j in range(0, image_width):
            if image[i][j] > threshold2:
                image[i][j] = 255
            else:
                image[i][j] = 0

    return image

# EXTENSION
# method to apply otsu's thresholding algorithm to convert the image to a binary image
def otsuThresholdEXTENSION(image, image_width, image_height):
    print("calculating otsu's threshold")
    # get histogram frequency for q by flattening the greyscale pixel array
    q = []
    flattenedImage = [element for element2 in image for element in element2]
    q = list(set(flattenedImage))

    # get histogram hq
    hq = []
    for b in (q):
        hq.append(flattenedImage.count(b))
        
    # get cumulative histogram cq
    cum = []
    for b in range(0,len(hq)):
        if b==0:
            cum.append(hq[0])
        else:
            cum.append(cum[b-1] + hq[b])

    # calculate qHq
    qHq = []
    
    for i in range(0,len(q)):
        qHq.append(q[i]*hq[i])

    # exhaustively calculate otsu's threshold for every histogram bin
    thresholds = []
    for threshold in q:
        hq_ob = []
        qhq_ob = []
        hq_bg = []
        qhq_bg = []

        # calculate average intensity for object and background
        for i in range(0,len(hq)):
            if q[i] >= threshold:
                hq_ob.append(hq[i])
                qhq_ob.append(q[i]*hq[i])
            else:
                hq_bg.append(hq[i])
                qhq_bg.append(q[i]*hq[i])

        # calculating otsu's threshold
        if threshold == 0:
            thresholds.append(0)
        else:
            wb_ob = sum(hq_ob)/image_width*image_height
            wf_back = sum(hq_bg)/image_width*image_height
            avg_ob = sum(qhq_ob)/sum(hq_ob)
            avg_bg = sum(qhq_bg)/sum(hq_bg)

            threshold2 = round(wb_ob*wf_back*((avg_ob-avg_bg)*(avg_ob-avg_bg)))
            thresholds.append(threshold2)

    # getting the threshold value in terms of q (histogram)
    max_threshold_otsu = max(thresholds)
    max_threshold = q[thresholds.index(max_threshold_otsu)]

    # apply threshold to the image
    for i in range(0, image_height):
        for j in range(0, image_width):
            if image[i][j] > max_threshold:
                image[i][j] = 255
            else:
                image[i][j] = 0

    return image

# method to apply a 5x5 dilation filter to the binary image with border zero padding
def dilateImage(pixel_array, image_width, image_height):
    print("dilating image")
    padding = createInitializedGreyscalePixelArray(image_width+4, image_height+4)
    padded = createInitializedGreyscalePixelArray(image_width+4, image_height+4)
    depadded = createInitializedGreyscalePixelArray(image_width, image_height)

    kernel = [[0, 0, 1, 0, 0],
              [0, 1, 1, 1, 0],
              [1, 1, 1, 1, 1],
              [0, 1, 1, 1, 0],
              [0, 0, 1, 0, 0]]
    
    # adding borderZeroPadding
    for i in range(0, image_height):
        for j in range(0, image_width):
            padding[i+2][j+2] = pixel_array[i][j]
    
    #dilate the image
    for i in range(2, image_height+2):
        for j in range(2, image_width+2):
            #if kernel hits in the image
            if padding[i-2][j] >=1 or padding[i-1][j-1] >=1 or padding[i-1][j] >=1 or padding[i-1][j+1] >=1 or padding[i][j-2] >=1 or padding[i][j-1] >=1 or padding[i][j] >=1 or padding[i][j+1] >=1 or padding[i][j+2] >=1 or padding[i+1][j-1] >=1 or padding[i+1][j] >=1 or padding[i+1][j+1] >=1 or padding[i+2][j] >=1:
                # 5x5 around i j is set to 255
                padded[i-2][j-2] = 255
                padded[i-2][j-1] = 255
                padded[i-2][j] = 255
                padded[i-2][j+1] = 255
                padded[i-2][j+2] = 255
                padded[i-1][j-2] = 255
                padded[i-1][j-1] = 255
                padded[i-1][j] = 255
                padded[i-1][j+1] = 255
                padded[i-1][j+2] = 255
                padded[i][j-2] = 255
                padded[i][j-1] = 255
                padded[i][j] = 255
                padded[i][j+1] = 255
                padded[i][j+2] = 255
                padded[i+1][j-2] = 255
                padded[i+1][j-1] = 255
                padded[i+1][j] = 255
                padded[i+1][j+1] = 255
                padded[i+1][j+2] = 255
                padded[i+2][j-2] = 255
                padded[i+2][j-1] = 255
                padded[i+2][j] = 255
                padded[i+2][j+1] = 255
                padded[i+2][j+2] = 255
                
    #removing borderZeroPadding
    for i in range(0, image_height):
        for j in range(0, image_width):
            depadded[i][j] = padded[i+2][j+2]
       
    return depadded

# method to apply a 5x5 erosion filter to a binary image with border zero padding
def erodeImage(pixel_array, image_width, image_height):
    print("eroding image")
    padding = createInitializedGreyscalePixelArray(image_width+4, image_height+4)
    paddedErosion = createInitializedGreyscalePixelArray(image_width+4, image_height+4)
    depadded = createInitializedGreyscalePixelArray(image_width, image_height)
    
    # adding borderZeroPadding
    for i in range(0, image_height):
        for j in range(0, image_width):
            padding[i+2][j+2] = pixel_array[i][j]
    
    #dilate the image
    for i in range(2, image_height+2):
        for j in range(2, image_width+2):
            #if kernel fits in the image
            if padding[i-2][j] >=1 and padding[i-1][j-1] >=1 and padding[i-1][j] >=1 and padding[i-1][j+1] >=1 and padding[i][j-2] >=1 and padding[i][j-1] >=1 and padding[i][j] >=1 and padding[i][j+1] >=1 and padding[i][j+2] >=1 and padding[i+1][j-1] >=1 and padding[i+1][j] >=1 and padding[i+1][j+1] >=1 and padding[i+2][j] >=1:
                paddedErosion[i][j] = 255
                
    #removing borderZeroPadding
    for i in range(0, image_height):
        for j in range(0, image_width):
            depadded[i][j] = paddedErosion[i+2][j+2]
                
    return depadded

def openingImageEXTENSION(image, image_width, image_height):
    print("opening image")
    eroded = erodeImage(image, image_width, image_height)
    dilated = dilateImage(eroded, image_width, image_height)

    return dilated

# performs connected component analysis on a binary image and returns the labels of the connected components
def connectedComponents(image, image_width, image_height):
    print("performing connected component analysis")
    labels = createInitializedGreyscalePixelArray(image_width, image_height)
    seen = createInitializedGreyscalePixelArray(image_width, image_height)
    label = 1

    for i in range(0, image_height):
        for j in range(0, image_width):

            # if pixel is not object and not seen
            if image[i][j] != 0 and seen[i][j] == 0:
                queue = []
                queue.append((i, j))
                
                # perform breadth first search while the queueueue is not empty
                while len(queue) > 0:

                    # get the pixel from the queue and mark as seen
                    (row, cols) = queue.pop()
                    labels[row][cols] = label
                    seen[row][cols] = 1

                    # check if the pixel is object and not seen and add to queue
                    if cols+1 < image_width and row+1 < image_height:
                        # check the right neighbour
                        if seen[row][cols+1] == 0 and image[row][cols+1] != 0:
                            queue.append((row, cols+1))
                            seen[row][cols+1] = 1
                        
                        # check the top neighbour
                        if seen[row-1][cols] == 0 and image[row-1][cols] != 0 and row-1 >= 0:
                            queue.append((row-1, cols))
                            seen[row-1][cols] = 1

                        # check the left neighbour
                        if seen[row][cols-1] == 0 and image[row][cols-1] != 0 and cols >= 0:
                            queue.append((row, cols-1))
                            seen[row][cols-1] = 1

                        # check the bottom neighbour
                        if seen[row+1][cols] == 0 and image[row+1][cols] != 0:
                            queue.append((row+1, cols))
                            seen[row+1][cols] = 1
                
                # increment label to be used for the next object
                label = label + 1

                # get list of different labels generated from the image
                uniqueLabels = []
                for i in range(0, image_height):
                    for j in range(0, image_width):
                        if labels[i][j] not in uniqueLabels:
                            uniqueLabels.append(labels[i][j])

                uniqueLabels.remove(0)
    
    # return the an image of objects with labels and a list of unique labels (number of objects)
    return labels, uniqueLabels
    
# method to find the min,max x and y values for each object in the image
def findBoundingboxDetails(labels, image_width, image_height, object_labels):
    print("finding bounding box details")
    boundingBoxLimits = []
    coinTypes = []
    coinNums = []

    # for the amount of objects in the image, get bounding box values
    for objects in range(0, len(object_labels)):

        limits = [0, 0, 0, 0]
        x = []
        y = []

        # find min and max x and y values for each object and average colour
        for i in range(0, image_height):
            for j in range(0, image_width):
                if labels[i][j] == object_labels[objects]:
                    x.append(j)
                    y.append(i)
        
        # if the object is large enough, add the bounding box values to the list
        if max(x)-min(x) > 180 and max(y)-min(y) > 180 and max(x)-min(x) < 300 and max(y)-min(y) < 300:        
            limits[0] = min(x)
            limits[1] = min(y)
            limits[2] = max(x)
            limits[3] = max(y)
            boundingBoxLimits.append(limits)

            #classifying objects based on average width
            width = max(x)-min(x)
            if width > 180 and width <= 215:
                coinTypes.append('10c')
                coinNums.append(0.1)
            elif width > 215 and width <= 236:
                coinTypes.append('20c')
                coinNums.append(0.2)
            elif width > 236 and width <= 250:
                coinTypes.append('$1')
                coinNums.append(1)
            elif width > 250 and width <= 268:
                coinTypes.append('50c')
                coinNums.append(0.5)
            elif width > 268 and width <= 285:
                coinTypes.append('$2')
                coinNums.append(2)
            else:
                coinTypes.append('Unknown')
        
    return [boundingBoxLimits, coinTypes, sum(coinNums)]


# This is our code skeleton that performs the coin detection.
def main(input_path, output_path):
    # This is the default input image, you may change the 'image_name' variable to test other images.
    image_name = 'hard_case_2'
    input_filename = f'./Images/hard/{image_name}.png'

    if TEST_MODE:
        input_filename = input_path

    # we read in the png file, and receive three pixel arrays for red, green and blue components, respectively
    # each pixel array contains 8 bit integer values between 0 and 255 encoding the color values
    (image_width, image_height, px_array_r, px_array_g, px_array_b) = readRGBImageToSeparatePixelArrays(input_filename)
    # computeRGBToGreyscale(px_array_r, px_array_g, px_array_b, image_width, image_height)

    
    # bounding_box_list = [[150, 140, 200, 190]]  # This is a dummy bounding box list, please comment it out when testing your own code.
    greyscaled = computeRGBToGreyscale(px_array_r, px_array_g, px_array_b, image_width, image_height)
    normalised = normaliseImage(greyscaled, image_width, image_height)
    laplaced = laplacianFilterEXTENSION(normalised, image_width, image_height)
    median3 = medianFilterEXTENSION(laplaced, image_width, image_height)
    # blurred = gaussianBlurEXTENSION(median3, image_width, image_height)               # uncomment to use experimental pipeline
    # blurred2 = gaussianBlurEXTENSION(blurred, image_width, image_height)              # uncomment to use experimental pipeline
    # blurred3 = gaussianBlurEXTENSION(blurred2, image_width, image_height)             # uncomment to use experimental pipeline
    blurred = blurImage(median3, image_width, image_height)                             # comment to use experimental pipeline
    blurred2 = blurImage(blurred, image_width, image_height)                            # comment to use experimental pipeline
    blurred3 = blurImage(blurred2, image_width, image_height)                           # comment to use experimental pipeline
    thresholded = adaptiveThresholdImageEXTENSION(blurred3, image_width, image_height)  # comment to use experimental pipeline
    # thresholded = otsuThresholdEXTENSION(blurred3, image_width, image_height)         # uncomment to use experimental pipeline
    dilated = dilateImage(thresholded, image_width, image_height)
    dilated2 = dilateImage(dilated, image_width, image_height)
    dilated3 = dilateImage(dilated2, image_width, image_height)
    eroded = erodeImage(dilated3, image_width, image_height)
    eroded2 = erodeImage(eroded, image_width, image_height)
    eroded3 = erodeImage(eroded2, image_width, image_height)
    eroded4 = erodeImage(eroded3, image_width, image_height)
    labels, uniqueLabels = connectedComponents(eroded4, image_width, image_height)
    bounding_box_list = findBoundingboxDetails(labels, image_width, image_height, uniqueLabels)[0]

    # px_array = thresholded
    px_array = pyplot.imread(input_filename)
    
    fig, axs = pyplot.subplots(1, 1)
    axs.imshow(px_array, aspect='equal')
    
    # Loop through all bounding boxes
    print("drawing bounding boxes")
    i = 0
    for bounding_box in bounding_box_list:
        details = findBoundingboxDetails(labels, image_width, image_height, uniqueLabels)
        coins = details[1]
        total = details[2]
        bbox_min_x = bounding_box[0]
        bbox_min_y = bounding_box[1]
        bbox_max_x = bounding_box[2]
        bbox_max_y = bounding_box[3]
        
        bbox_xy = (bbox_min_x, bbox_min_y)
        bbox_width = bbox_max_x - bbox_min_x
        bbox_height = bbox_max_y - bbox_min_y
        rect = Rectangle(bbox_xy, bbox_width, bbox_height, linewidth=2, edgecolor='r', facecolor='none')
        axs.add_patch(rect)
        # adding text to the bounding box
        text_x = bbox_min_x
        text_y = bbox_min_y + 10
        axs.text(text_x, text_y, f"Coin: {(details[1])[i]}\n", fontsize=11, color='r')
        i += 1
    print("Number of coins: ",len(coins))
    text_x = 18
    text_y =  image_height + 10
    axs.text(text_x, text_y, f"Number of coins: {len(coins)}  Total amount: ${total}\n", fontsize=11, color='r')

    pyplot.axis('off')
    pyplot.tight_layout()
    default_output_path = f'./output_images/{image_name}_with_bbox.png'
    if not TEST_MODE:
        # Saving output image to the above directory
        pyplot.savefig(default_output_path, bbox_inches='tight', pad_inches=0)
        
        # Show image with bounding box on the screen
        pyplot.imshow(px_array, cmap='gray', aspect='equal')
        pyplot.show()
    else:
        # Please, DO NOT change this code block!
        pyplot.savefig(output_path, bbox_inches='tight', pad_inches=0)



if __name__ == "__main__":
    num_of_args = len(sys.argv) - 1
    
    input_path = None
    output_path = None
    if num_of_args > 0:
        input_path = sys.argv[1]
        output_path = sys.argv[2]
        TEST_MODE = True
    
    main(input_path, output_path)
    