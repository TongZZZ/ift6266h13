import numpy 
import numpy.random
import pylab

from dispims import dispims as dispims_gray
import online_kmeans
import theano
import theano.tensor as T

def get_most_square_shape(n):
    mostSquareShape = [1,n]
    for i in range(2,int(n**0.5)+1):
        if n % i == 0:
            mostSquareShape = [i,n/i]
    return mostSquareShape
    
def pca(data, var_fraction):
    """ principal components, retaining as many components as required to 
        retain var_fraction of the variance 

    Returns projected data, projection mapping, inverse mapping, mean"""
    from numpy.linalg import eigh
    u, v = eigh(numpy.cov(data, rowvar=1, bias=1))
    v = v[:, numpy.argsort(u)[::-1]]
    u.sort()
    u = u[::-1]
    u = u[u.cumsum()<u.sum()*var_fraction]
    numprincomps = u.shape[0]
    V = ((u**(-0.5))[:numprincomps][numpy.newaxis,:]*v[:,:numprincomps]).T
    W = (u**0.5)[:numprincomps][numpy.newaxis,:]*v[:,:numprincomps]
    return numpy.dot(V,data), V, W
 
#********************** 

def getTheanoConvFunction(patchsize=None, imagesize=None):
    """
    Return a theano function erforming valid convolution of a filter on an
    image
    """
    
    # Define the size of the images and filters to allow Theano to
    # further optimize the convolution op
    image_shape = (None, 1, imagesize, imagesize)
    filter_shape = (None, 1, patchsize, patchsize)
    
    # Define the input variables to the function
    img = T.tensor4(dtype='floatX')
    filter = T.tensor4(dtype='floatX')
    mask = T.tensor4(dtype='floatX')
    
    # Convolve the image with both the filter and the mask
    convImgWithFilter = T.nnet.conv.conv2d(img, filter, border_mode='valid',
                                           image_shape=image_shape,
                                           filter_shape=filter_shape)
     
    # Norm convImgWithFilter by the norm of each portions of the image's norm
    # to avoid a brighter region taking the lead on a darker, better-fitting
    # one.                                      
    convImgWithMask = T.nnet.conv.conv2d(img**2, mask, border_mode='valid',
                                         image_shape=image_shape,
                                         filter_shape=filter_shape)
    convImgWithMask = convImgWithMask ** 0.5
    
    normConvImgWithFilter = convImgWithFilter / (convImgWithMask ** 0.5)
     
    # Compile and return the theano function
    f = theano.function([img, filter, mask], normConvImgWithFilter)
    return f
    
def getTheanoSimilarityFunction():
    """
    Return a theano function erforming valid convolution of a filter on an
    image
    """
        
    # Define the input variables to the function
    patches = T.tensor3(dtype='float32') # AxBx(patchsize**2)
    filters = T.matrix(dtype='float32') # Cx(patchsize**2)
    globalMean = T.vector(dtype='float32')
    globalStd = T.vector(dtype='float32')
    
    # Perform canonical processing of the patches
    meanstd = patches.std()
    mean = T.shape_padright(patches.mean(2), n_ones=1)
    std = T.shape_padright(patches.std(2) + 0.1 * meanstd, n_ones=1)  
    std = T.shape_padright(patches.std(2) + 1e-6, n_ones=1)  
    canonicalPatches_ = (patches - mean) / std  
    canonicalPatches = (canonicalPatches_ - globalMean) / globalStd  

    # Compute the similarities between each patch and each filter
    similarities = T.tensordot(canonicalPatches, filters, axes=[[2],[1]]) # AxBxC
    
    normFactor = ((canonicalPatches** 2).sum(2) ** 0.5)
    normFactorPadded = T.shape_padright(normFactor, n_ones=1)
    
    # Normalize the similarities by the norm of the patches
    similaritiesNorm = (similarities / normFactorPadded)
    
    # Compile and return the theano function
    f = theano.function([patches, filters, globalMean, globalStd], 
                        similaritiesNorm, on_unused_input='ignore')
    return f

simFunction = getTheanoSimilarityFunction()


def getGaussianHeatmap(shape, meanRow, stdRow, meanCol, stdCol):
    import matplotlib.mlab as mlab
    heatmap = numpy.zeros(shape, dtype='float32')
    trunkPoint = 0.20 
    for r in range(shape[0]):
        r_pdf = mlab.normpdf(r, meanRow, stdRow)
        for c in range(shape[1]):
            c_pdf = mlab.normpdf(c, meanCol, stdCol)
            heatmap[r,c] = r_pdf * c_pdf

            if (r<meanRow-stdRow*trunkPoint or r>meanRow+stdRow*trunkPoint or
                c<meanCol-stdCol*trunkPoint or c>meanCol+stdCol*trunkPoint):
                heatmap[r,c]=0.0
    
    return heatmap


def extractPatchMap(image, filterShape):
    nbPixels = filterShape[0] * filterShape[1]
    patches_ = numpy.zeros((96, 96, nbPixels), dtype='float32')
    
    # Extract all possible patches from the image
    paddedImage = numpy.zeros((image.shape[0]+(filterShape[0]/2)*2,
                               image.shape[1]+(filterShape[1]/2)*2,), dtype='float32')
    paddedImage[filterShape[0]/2:image.shape[0]+filterShape[0]/2,
                filterShape[1]/2:image.shape[0]+filterShape[1]/2] = image
    for r in range(image.shape[0]):
        for c in range(image.shape[1]):            
            patch = paddedImage[r:r+filterShape[0], c:c+filterShape[1]]
            patches_[r,c] = patch.reshape(nbPixels)
            
    return patches_
    

def getKeypointHeatMap(patchesMap, filters, canonicalData, pcaData, 
                       pooling):
    
    # Ensure the pooling type issupported
    assert (pooling == 'max' or pooling == 'mean')
    
    # TODO : make it work even near the borders
    # TODO : Ensure it works with even and odd patchsize

    # Normalize the patches
    similarities = simFunction(patchesMap,
                               filters.reshape(len(filters), filters.size/len(filters)),
                               canonicalData[0][0], canonicalData[1][0])
                       
    convMaps = numpy.array(similarities).transpose(2,0,1)
        
    # Combine the convMaps using the specified pooling
    if pooling == "max":
        pooledConvMap = convMaps.max(0)
    elif pooling == "mean":
        pooledConvMap = convMaps.mean(0)
        
    # Compute the softmax of the pooledConvMap to get a heatmap for the
    # keypoint
    heatmap = softmax(pooledConvMap)
        
    # Combine the heatmaps using the specified pooling
    return heatmap


def softmax(array):

    expArray = numpy.exp(array)
    softmaxOutput = expArray / expArray.sum()
    return softmaxOutput 

def predictFromHeatmaps(heatmaps):
    predictions = []
    for heatmap in heatmaps:
        prediction = numpy.unravel_index(heatmap.argmax(), heatmap.shape)
        predictions.append(prediction[1]) 
        predictions.append(prediction[0])
    return predictions
    
def predictFromHeatmaps2(heatmaps1, heatmaps2):
    
    predictions = []
    for i in range(len(heatmaps1)):
        totalHeatmap = numpy.maximum(heatmaps1[i], heatmaps2[i])
        prediction = numpy.unravel_index(totalHeatmap.argmax(), totalHeatmap.shape)
        predictions.append(prediction[1])  
        predictions.append(prediction[0])
    return predictions
    
def iround(num):
    if (num > 0):
        return int(num+.5)
    else:
        return int(num-.5)    

def canonical(patches, globalMean=None, globalStd=None): 
    """  
    meanstd = patches.std()
    patches -= patches.mean(1)[:,None]
    patches /= patches.std(1)[:,None] + 0.1 * meanstd
    if globalMean == None:
        globalMean = patches.mean(0)[None,:]
    if globalStd == None:
        globalStd = patches.std(0)[None,:]
    patches -= globalMean
    patches /= globalStd
    
    return (patches, globalMean, globalStd)

    """
    patchesCP = patches.copy()

    meanstd = patchesCP.std()
    patchesCP -= patchesCP.mean(1)[:,None]
    patchesCP /= patchesCP.std(1)[:,None] + 1e-6 #0.1 * meanstd
    if globalMean == None:
        globalMean = patchesCP.mean(0)[None,:]
    if globalStd == None:
        globalStd = patchesCP.std(0)[None,:]
    patchesCP -= globalMean
    patchesCP /= globalStd
    
    return (patchesCP, globalMean, globalStd)
    

def pca(data, var_fraction):
    """ principal components, retaining as many components as required to 
        retain var_fraction of the variance 

    Returns projected data, projection mapping, inverse mapping, mean"""
    from numpy.linalg import eigh
    u, v = eigh(numpy.cov(data, rowvar=1, bias=1))
    v = v[:, numpy.argsort(u)[::-1]]
    u.sort()
    u = u[::-1]
    u = u[u.cumsum()<u.sum()*var_fraction]
    numprincomps = u.shape[0]
    V = ((u**(-0.5))[:numprincomps][numpy.newaxis,:]*v[:,:numprincomps]).T
    W = (u**0.5)[:numprincomps][numpy.newaxis,:]*v[:,:numprincomps]
    return numpy.dot(V,data), V, W

#***********************

# Define the constants to be used throughout this pipeline
imagesize = 96
patchsize = 7
nbClusters = 100
validStart = 6500
nbImages = 7049
displayImages = False
performValid = True
generateSubmission = False
kaggleSubmissionFile = "kaggleSubmission.csv"
featureNames = ["left_eye_center_x","left_eye_center_y",
                    "right_eye_center_x","right_eye_center_y",
                    "left_eye_inner_corner_x","left_eye_inner_corner_y",
                    "left_eye_outer_corner_x","left_eye_outer_corner_y",
                    "right_eye_inner_corner_x","right_eye_inner_corner_y",
                    "right_eye_outer_corner_x","right_eye_outer_corner_y",
                    "left_eyebrow_inner_end_x","left_eyebrow_inner_end_y",
                    "left_eyebrow_outer_end_x","left_eyebrow_outer_end_y",
                    "right_eyebrow_inner_end_x","right_eyebrow_inner_end_y",
                    "right_eyebrow_outer_end_x","right_eyebrow_outer_end_y",
                    "nose_tip_x","nose_tip_y",
                    "mouth_left_corner_x","mouth_left_corner_y",
                    "mouth_right_corner_x","mouth_right_corner_y",
                    "mouth_center_top_lip_x","mouth_center_top_lip_y",
                    "mouth_center_bottom_lip_x","mouth_center_bottom_lip_y"]

# Init the random number generator with a fixed seed for reproducibility
rng = numpy.random.RandomState(1)
    
# Load the training data and labels
from keypoints_dataset import FacialKeypointDataset
from pylearn2.datasets.preprocessing import ShuffleAndSplit

trainSet = FacialKeypointDataset(
                which_set='train', 
                preprocessor=ShuffleAndSplit(42, 0, validStart))
trainData = trainSet.X.reshape(validStart, imagesize, imagesize).astype("float32")
trainLbls = trainSet.y

# Compute the mean and standard deviation for each keypoint coordinate
nbTrainingCaseByCoord = (trainLbls != -1).sum(0)
nbTrainingCaseByKeypoint = (trainLbls != -1).sum(0)[::2]

meanCoords = (trainLbls * (trainLbls!=-1)).sum(0) / nbTrainingCaseByCoord
stdCoords  = (((trainLbls-meanCoords) ** 2) * (trainLbls!=-1)).sum(0) / nbTrainingCaseByCoord


stdCoords /= 1


# Normalize the standard deviations so that keypoints for which we have
# less examples have smaller standard deviation
#stdCoords /= stdNormFactors

# Normalize the number of clusters by keypoints to that keypoints with less
# training data for them have more clusters
#nbTrainingCaseByKeypoints = (trainLbls != -1).sum(0)[::2]
#nbClustersByKeypoint = [int(nbClusters * 1.0 * len(trainLbls) / 
#                            nbTrainingCaseByKeypoints[i])
#                        for i in range(trainLbls.shape[1]/2)]

# Compute the kmeans centroids for each keypoint
centroidsByKeypoint = []
canonicalDataByKeypoint = []
pcaDataByKeypoint = []
for keypointIdx in range(trainLbls.shape[1] / 2):
     
    # Compute the indices of the coordinates of the current keypoint in
    # the trainLbls array
    keypointIdxX = keypointIdx * 2
    keypointIdxY = keypointIdx * 2 + 1
    
    # Extract all the patches from the training data centered around the target
    # keypoint
    patches = []
    for imgIdx in range(len(trainData)):
        
        # Obtain the x and y coordinates of the keypoint in the image
        coordX = iround(trainLbls[imgIdx, keypointIdxX])
        coordY = iround(trainLbls[imgIdx, keypointIdxY])
        
        # Determine it we have the location of the current keypoint in the
        # current image        
        haveKeypointLocation = (coordX != -1 and coordY != -1)
                     
        if haveKeypointLocation:
            
            # Extract a patch centered around the keypoint coodinates
            # TODO : Ensure it works with odd and even patchsize
            # TODO : Make the script work for keypoints at the border of the images
            patch = trainData[imgIdx,
                              coordY-patchsize/2:coordY-patchsize/2+patchsize,
                              coordX-patchsize/2:coordX-patchsize/2+patchsize]
            
            # Ensure that the patch has the correct size
            if (patch.shape[0] == patchsize and
                patch.shape[1] == patchsize):
                    
                # Add the patch to the list
                patches.append(patch)
            
    # Cast the list of patches to a numpy array and reshape the patches for
    # easier and faster processing
    patches = numpy.array(patches).reshape(len(patches), patchsize ** 2)
    
    # Perform canonical preprocessing of the patches
    canonicalPatches, globalMean, globalStd = canonical(patches)
    canonicalDataByKeypoint.append((globalMean, globalStd))
    
    # Perform PCA-whitening preprocessing of the patches
    pcadata, pcaBackward, pcaForward = pca(canonicalPatches.T, .9)
    whitePatches = pcadata.T.astype("float32")
    pcaDataByKeypoint.append((pcaBackward, pcaForward))
    
    # Perform kmeans clustering of the patches
    Rinit = rng.permutation(nbClusters)
    W = whitePatches[Rinit]
    print "training kmeans"
    for epoch in range(50):
        W = online_kmeans.kmeans(whitePatches, nbClusters, Winit=W,
                                 numepochs=1, learningrate=0.01*0.8**epoch)
        
    # Project the centroids back in pixel space
    W_ = numpy.dot(pcaForward,W.T).T.reshape(nbClusters, patchsize, patchsize).astype("float32")
        
    # Display the centroids learned with kmeans
    if displayImages:
        f1 = pylab.figure()
        dispims_gray(W_[None,:,:,:], patchsize, patchsize)     
          
    # Add the kmeans centroids to the list of learned filters
    centroidsByKeypoint.append(W_)           


if performValid:

    # Load the validation data
    validSet = FacialKeypointDataset(
                    which_set='train', 
                    preprocessor=ShuffleAndSplit(42, validStart, nbImages))
    validData = validSet.X.reshape(nbImages - validStart, imagesize, imagesize).astype("float32")
    validLbls = validSet.y
    
    
   # Define the ponderations of the k-means heatmaps and the prior heatmap for each keypoint

    # Test the filters learned on the validation data
    squaredErrors = []
    for imgIdx in range(len(validData)):
        
        image = validData[imgIdx]
        label = validLbls[imgIdx]
        
        # Extract all the patches of the current image
        patchesMap = extractPatchMap(image, (patchsize, patchsize))
        
        def dispMap(heatmap):
            import Image
            Image.fromarray(heatmap).show()
            
        def saveArray(arr, filename):
            import Image
            Image.fromarray(arr).convert("L").save(filename)
            
        #Image.fromarray(getKeypointHeatMap(patchesMap, centroidsByKeypoint[1],canonicalDataByKeypoint[1], pcaDataByKeypoint[1],pooling="max")*500000).show()
        
        if imgIdx == 0:
            saveArray(image, "kpOriginal.jpg")
        
        # Compute a heatmap for each keypoint
        heatmaps = []
        for keypointIdx in range(validLbls.shape[1] / 2):
            heatmap = getKeypointHeatMap(patchesMap, centroidsByKeypoint[keypointIdx],
                                        canonicalDataByKeypoint[keypointIdx],
                                        pcaDataByKeypoint[keypointIdx],
                                        pooling="mean")
            heatmaps.append(heatmap)
             
            if imgIdx == 0:
                saveArray(heatmap/heatmap.max()*255, "kpHeatmap%i.jpg" % keypointIdx)
        
        # For each keypoint, compute a heat map indicating the prior
        # that we have about the keypoint location
        priorHeatmaps = []
        for keypointIdx in range(validLbls.shape[1] / 2):
            heatmap = getGaussianHeatmap(image.shape,
                                        meanCoords[keypointIdx*2+1], stdCoords[keypointIdx*2+1],
                                        meanCoords[keypointIdx*2], stdCoords[keypointIdx*2],
                                        )
            priorHeatmaps.append(heatmap)
            
            if imgIdx == 0:
                saveArray(heatmap/heatmap.max()*255, "kpPriorHeatmap%i.png" % keypointIdx)
        
        for keypointIdx in range(validLbls.shape[1] / 2):
            if imgIdx ==0:
                saveArray((heatmaps[keypointIdx] * priorHeatmaps[keypointIdx]) / 
                          (heatmaps[keypointIdx] * priorHeatmaps[keypointIdx]).max() * 255,
                          "kpCombinedHeatmap%i.png" % keypointIdx) 
            
        # From the heatmaps, perform a prediction of the location of each of
        # the keypoints
        kagglePrediction = predictFromHeatmaps([(heatmaps[keypointIdx])*(priorHeatmaps[keypointIdx])
                                       for keypointIdx in range(validLbls.shape[1] / 2)])
        #kagglePrediction = predictFromHeatmaps2([1.5*h for h in heatmaps], priorHeatmaps)
        #kagglePrediction = predictFromHeatmaps(heatmaps)  
        
        # Compute the mean squared error on the predicted keypoints locations
        squaredError = ((kagglePrediction - validLbls[imgIdx])**2 * 
                        (validLbls[imgIdx] != -1)).mean()
        squaredErrors.append(squaredError)

        print imgIdx, len(validData)
        print squaredError   
        print numpy.array(squaredErrors).mean()
        
        if imgIdx == 100:
            print kagglePrediction
            print validLbls[imgIdx]
            print kagglePrediction - validLbls[imgIdx]
            import pdb
            pdb.set_trace()

    print "done"
    print numpy.array(squaredErrors).mean()




if generateSubmission:

    # Load the test data
    testSet = FacialKeypointDataset(
                    which_set='public_test', 
                    preprocessor=ShuffleAndSplit(42, 0, 100000))
    testData = testSet.X.reshape(len(testSet.X), imagesize, imagesize).astype("float32")

    # Init the submission file
    f = open(kaggleSubmissionFile, "w")
    f.write("RowId,ImageId,FeatureName,Location\n")
    f.close()
    nextRowId = 1
    
    # Load the submission format file
    f = open("submissionFileFormat.csv", "r")
    submissionLines = f.readlines()
    submissionEntries = [line.strip().split(",") for line in submissionLines]
    f.close()

    # Test the filters learned on the validation data
    for imgIdx in range(len(testData)):
        print imgIdx, len(testData)

        image = testData[imgIdx]
        
        # Extract all the patches of the current image
        patchesMap = extractPatchMap(image, (patchsize, patchsize))
        
        # Compute a heatmap for each keypoint
        heatmaps = []
        for keypointIdx in range(trainLbls.shape[1] / 2):
            heatmap = getKeypointHeatMap(patchesMap, centroidsByKeypoint[keypointIdx],
                                        canonicalDataByKeypoint[keypointIdx],
                                        pcaDataByKeypoint[keypointIdx],
                                        pooling="max")
            heatmaps.append(heatmap)
        
        
        # For each keypoint, compute a heat map indicating the prior
        # that we have about the keypoint location
        priorHeatmaps = []
        for keypointIdx in range(trainLbls.shape[1] / 2):
            heatmap = getGaussianHeatmap(image.shape,
                                        meanCoords[keypointIdx*2], stdCoords[keypointIdx*2],
                                        meanCoords[keypointIdx*2+1], stdCoords[keypointIdx*2+1],
                                        )
            priorHeatmaps.append(heatmap)
        
            
        # From the heatmaps, perform a prediction of the location of each of
        # the keypoints
        predictions = predictFromHeatmaps([heatmaps[keypointIdx]*priorHeatmaps[keypointIdx] ** 0.3
                                        for keypointIdx in range(trainLbls.shape[1] / 2)])
        
        # Generate the kaggle submission row for that image
        kagglePrediction = []
        for kp in predictions:
            kagglePrediction.append(kp[1])
            kagglePrediction.append(kp[0])
        
        f = open(kaggleSubmissionFile, "a")
        for featureIdx in range(len(kagglePrediction)):
            if (submissionEntries[nextRowId][0] == str(nextRowId) and
                submissionEntries[nextRowId][1] == str(imgIdx+1) and
                submissionEntries[nextRowId][2] == featureNames[featureIdx]):
                       
                f.write("%i,%i,%s,%f\n" % (nextRowId, imgIdx+1,
                                           featureNames[featureIdx],
                                           kagglePrediction[featureIdx]))
                                    
                nextRowId += 1
            
        f.close() 


    print "Kaggle submission has been generated"

    
    
    
    
    
    
    
    
