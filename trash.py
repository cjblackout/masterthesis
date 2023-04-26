def cal_skyline(mask):
    h, w = mask.shape
    for i in range(w):
        raw = mask[:, i]
        after_median = medfilt(raw, 19) #perform median filter on the column
        try:
            first_zero_index = np.where(after_median == 0)[0][0] #get the index of the first zero value in the column
            first_one_index = np.where(after_median == 1)[0][0] #get the index of the first one value in the column
            if first_zero_index > 20:  #find the region between first zero and first one and paint it 1
                mask[first_one_index:first_zero_index, i] = 1
                mask[first_zero_index:, i] = 0
                mask[:first_one_index, i] = 0
        except:
            continue
        
    return mask


def get_sky_region_gradient(img, ksize=3, threshold=9):

    h, w, _ = img.shape
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #convert to grayscale
    img_gray = cv2.blur(img_gray, (9, 3)) #blur the image to reduce noise
    cv2.medianBlur(img_gray, 5) #blue with median filter

    #lap = cv2.Laplacian(img_gray, cv2.CV_8U) #get the laplacian of the image

    #get the sobel operator in x and y direction and take L2 norm of the two
    sobelx = cv2.Sobel(img_gray, cv2.CV_8U, 1, 0, ksize=ksize)
    sobely = cv2.Sobel(img_gray, cv2.CV_8U, 0, 1, ksize=ksize)
    sobel = np.float32(np.sqrt(sobelx ** 2 + sobely ** 2))

    #print("Sobel Operator Output: ")
    #plt.figure(figsize=(20, 20))
    #plt.imshow(sobel)
    #plt.show()
    
    gradient_mask = (sobel < threshold).astype(np.uint8) #get the gradient mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)) #create a kernel for morphological operations

    mask = cv2.morphologyEx(gradient_mask, cv2.MORPH_ERODE, kernel) #perform Morphological Erosion
    mask = cal_skyline(mask) #calculate the skyline of the image
    #mask_inverted = 1 - mask #invert the mask

    after_img = cv2.bitwise_and(img, img, mask=mask) #perform bitwise and to apply the mask from cal_skylines

    """#get the average color of after_img
    avg_color_per_row = np.average(after_img, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)

    #calculate the standard deviation of after_img
    std_color_per_row = np.std(after_img, axis=0)
    std_color = np.std(std_color_per_row, axis=0)

    #calculate the probability distribution of the image standardised with avg_color and std_color
    prob_color = np.exp(-1*((img - avg_color) ** 2) / (5 * std_color ** 2))
    prob_color = np.sum(prob_color, axis=2)   #check for probability 
    prob_color = 1 - (prob_color / np.sum(prob_color)) #normalize the probability distribution
    #prob_color = (prob_color - np.min(prob_color)) / (np.max(prob_color) - np.min(prob_color)) #scale between 0 and 1

    #NOTE: Clearly not gaussian as 1 - prob_color != inverted(prob_color)

    #calculate the exponential probability distribution of gradient_mask
    prob_gradient = np.exp(-1*((1 - gradient_mask) ** 2))
    #prob_gradient = (prob_gradient - np.min(prob_gradient)) / (np.max(prob_gradient) - np.min(prob_gradient)) #scale between 0 and 1

    #calculate the exponential probability of vertical pixel value divided by total height for the width of the image
    prob_vertical = np.tile(np.exp(-1*((np.arange(h) / h) ** 2)), (w, 1)).T
    #prob_vertical = (prob_vertical - np.min(prob_vertical)) / (np.max(prob_vertical) - np.min(prob_vertical)) #scale between 0 and 1

    #show the prob_dist as image
    #print("Exponential Distribution of the color of the image based on the average color and standard deviation of the sky region: ")
    #print(prob_color)
    #plt.figure(figsize=(20, 20))
    #plt.imshow(prob_color)
    #plt.show()


    #show prob_gradient as image
    #print("Expontential Distribution of the gradient of the image:")
    #print(prob_gradient)
    #plt.figure(figsize=(20, 20))
    #plt.imshow(prob_gradient)
    #plt.show()

    #show prob_vertical as image
    #print("Exponential Distribution of the vertical pixel value:")
    #print(prob_vertical)
    #plt.figure(figsize=(20, 20))
    #plt.imshow(prob_vertical)
    #plt.show()

    #multiply the three probability distributions
    prob_dist = prob_color * prob_vertical * prob_gradient

    #show the final probability distribution as image
    #print("Final Probability Distribution:")
    #print(prob_dist)
    #plt.figure(figsize=(20, 20))
    #plt.imshow(prob_dist)
    #plt.show()

    #show image with mask applied
    #print("Image with mask applied:")
    #plt.figure(figsize=(20, 20))
    #plt.imshow(cv2.bitwise_and(img, img, mask=mask))
    #plt.show()

    #create a mask from the probability distribution by thresholding it with 1 * 10^-5
    prob_mask = (prob_dist > 6 * 10 ** -1).astype(np.uint8)

    after_img = cv2.bitwise_and(img, img, mask=prob_mask)"""

    prob_mask = mask

    return after_img, mask, prob_mask

def multi_task(file):
    img = cv2.imread(file)[:,:,::-1]
    val_image = cv2.imread("C:\\Users\\cjbla\\OneDrive\\Desktop\\Code\\data\\dataset\\ValidationImages\\Skyfinder\\" + file.split("\\")[-2] + ".png")
#print("Original Image:")
#plt.figure(figsize=(20, 20))
#plt.imshow(img)
#plt.show()

    _, mask_3_9, _ = get_sky_region_gradient(img, 3, 9)
    _, mask_5_9, _ = get_sky_region_gradient(img, 5, 9)
    _, mask_7_9, _ = get_sky_region_gradient(img, 7, 9)
#print("Image with sky region removed:")
#plt.figure(figsize=(20, 20))
#plt.imshow(img_sky)
#plt.show()\


#print("Ground Truth Image:")
#plt.figure(figsize=(20, 20))
#plt.imshow(val_image)
#plt.show()

#calculate the accuracy of the mask
#print("Accuracy of the mask: " + str(np.sum(mask == val_image[:,:,0]) / (val_image.shape[0] * val_image.shape[1])))
#print("Accuracy of the prob_mask: " + str(np.sum(prob_mask == val_image[:,:,0]) / (val_image.shape[0] * val_image.shape[1])))
    print("file: " + str(file.split("\\")[-2]))

#append the accuracy values in a 2D array
    return [np.sum(mask_3_9 == val_image[:,:,0]) / (val_image.shape[0] * val_image.shape[1]), np.sum(mask_5_9 == val_image[:,:,0]) / (val_image.shape[0] * val_image.shape[1]), np.sum(mask_7_9 == val_image[:,:,0]) / (val_image.shape[0] * val_image.shape[1])]

    #print("iterations: " + str(len(acc)))

acc = []
files = [os.path.join(dp, f) for dp, dn, filenames in os.walk("C:\\Users\\cjbla\\OneDrive\\Desktop\\Code\\Thesis\\dataset\\OriginalImages") for f in filenames if os.path.splitext(f)[1] == '.jpg']
for file in files:
    acc.append(multi_task(file))

#sort acc
acc.sort(key=lambda x: x[1])

#plot acc as a line graph with the x-axis being the number of iterations and the y-axis being the accuracy
plt.figure(figsize=(20, 20))
plt.plot(np.arange(len(acc)), np.array(acc)[:,0], label="Mask_3_9")
plt.plot(np.arange(len(acc)), np.array(acc)[:,1], label="Mask_5_9")
plt.plot(np.arange(len(acc)), np.array(acc)[:,2], label="Mask_7_9")
plt.legend()
plt.show()

#save acc as a csv file
np.savetxt("C:\\Users\\cjbla\\OneDrive\\Desktop\\Code\\Thesis\\dataset\\acc.csv", np.array(acc), delimiter=",")