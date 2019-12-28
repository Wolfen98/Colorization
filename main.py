import copy #using deepcopy, since copy only passes on the reference
import numpy as np
import cv2
import os
from sklearn.preprocessing import StandardScaler
from PIL import Image #errors with just importing PIL, Attribute Error

class NeuralNetwork():
    
    rectifiedLinear = 1 #main activation function to use for hidden lengthLayers. Fast and efficient
    # intializes the neural network with specified attributes
    def __init__(self, epochs = None, hiddenNeurons = None, hiddenLayers = None, learningRate = None, batches = None, weights = None):
        self.epochs = epochs #entire dataset given a feed forward and backwardprop. Need to divide into batches to increase performance and prevent memory errors
        self.hiddenNeurons = hiddenNeurons # amount of hidden neurons per each layer
        self.hiddenLayers = hiddenLayers # lengthLayers between input and output layer
        self.layers = self.hiddenLayers + 1 #weight array with the bias
        self.learningRate = learningRate #standard learning rate is .003, which I will use
        self.batches = batches # splits up epoch into iterations. 
        self.weights = weights
        # I will do a simple ReLu calcuation for every activation layer in the neural network. Does not suffer from disappearing gradient
        self.activations = [self.rectifiedLinear] + ([self.rectifiedLinear] * self.hiddenLayers) #Apply the activation function to every hidden layer, and then implement
        self.functions = {self.rectifiedLinear : self.calcReLu}
        self.derivatives = {self.rectifiedLinear: self.calcReluDerivative}

    
    def createWeights(self, num):
        self.weights = []
        self.hiddenNeurons.append(1)
        self.lengthLayers = range(self.layers)
        # Creating an array of neurons with associated weights, randomized towards the distribution of the Gaussian curve
        for layer in self.lengthLayers:
            self.weights.append([])
            neurons = self.hiddenNeurons[layer]
            if (layer == 0):
                self.weights[layer] = np.random.normal(scale = 1, size = (neurons, num))
            else:
                # For the rest of the neurons, do the same, and apply the bias neuron
                self.weights[layer] = np.random.normal(scale = 1, size = (neurons, 1 + self.hiddenNeurons[layer - 1]))

        self.weights = np.array(self.weights)
        self.outdatedWeights = copy.deepcopy(self.weights)

    #main function for the feed forward, run the array of RGB each through the neural network    
    def feedForward(self, num):
        result = [] #initialize the outputs array 
        for x in self.lengthLayers:
            result.append([]) 
            activation = self.activations[x]
            function = self.functions[activation]
            #Perform the calculation on the first layer, and then do th rest of the neural network web
            if (x == 0):
                curr = num
            else:
                curr = result[x - 1].copy()
                curr = np.insert(curr, obj = 0, values = 1)

            result[x] = function(np.matmul(self.weights[x], curr))

        result = np.array(result)
        return result

    def backwardsProp(self, row, col, result):

        # The derivative array will be the output of backwards propagation. Will be used to update weights
        derivatives = copy.deepcopy(result)
        bpLayers = self.lengthLayers[::-1]
        wDerivatives = copy.deepcopy(self.weights)

        # First, compute the derivatives backwards before updating them. Calculate the activation derivation with the bias
        #and then calculate the dot product of the derivative activation layer and the new weights that go along with it.
        #Then 
        for x in bpLayers:
            colorArr = x + 1
            if x == self.layers - 1: #for the last layer
                derivatives[x] = 2*(result[x] - col)
                continue

            next = self.activations[colorArr]
            dLayer = self.derivatives[next]
            dNextLayer = derivatives[colorArr]
            curr = result[x].copy()
            curr = np.insert(curr, obj = 0, values = 1)
            nextdLayer = dLayer(np.matmul(self.outdatedWeights[colorArr], curr))#computing derivatives 
            nextdLayer = nextdLayer.reshape(-1, 1) #To prevent illegal matrix multiplication errors
            weightLayer = self.outdatedWeights[colorArr][:, 1:] #remove the bias before multiplying the next derivative layer without bias
            thisLayer = nextdLayer * weightLayer
            derivatives[x] = np.matmul(dNextLayer, thisLayer)

        #Now to update all the weights with the designated output of derivatives. Retrieve the current layer and then multiply with every neuron in th next layers
        for x in bpLayers:
            currLayer = self.activations[x]
            dLayer = self.derivatives[currLayer]
            if x == 0: #if its the first layer
                thisLayer = row
            else:
                prev_layer = x - 1
                thisLayer = result[prev_layer].copy()
                thisLayer = np.insert(thisLayer, obj = 0, values = 1)

            output = derivatives[x].reshape(-1, 1) #preventing errors in mulitplication
            updatedDLayer = dLayer(np.matmul(self.outdatedWeights[x],thisLayer))
            updatedDLayer = updatedDLayer.reshape(-1, 1)
            currDLayer = output * updatedDLayer * thisLayer #applying calcualtion to every neutron in current layer
            wDerivatives[x] = currDLayer

        # Append the current data point's weight derivatives in the batch derivatives array
        self.updatedWeights.append(wDerivatives)

    def fit(self, trainData, colorArr):
        #fitting the neural network data to match the training data with the extracted RGB arrays from set of images. Uses Back propation to better its estimates
        #Loss is always high. This may mean that the ReLu model isn't working with my image classification
        newTrainData = np.column_stack((np.ones(len(trainData)), trainData))
        self.createWeights(newTrainData.shape[1])

        for epoch in range(self.epochs):
            self.updatedWeights = []
            for x in range(newTrainData.shape[0]):
                result = self.feedForward(newTrainData[x])#inserting thhe next set of training data to the weighted neurons
                self.backwardsProp(newTrainData[x], colorArr[x], result) #adjusting weights with the actual color array values
                if (x + 1) % self.batches == 0: #updating the weights as the fit goes along, part of back propogation
                    dbatches = np.mean(self.updatedWeights, axis = 0)
                    self.weights = self.outdatedWeights - (self.learningRate * dbatches)
                    self.outdatedWeights = copy.deepcopy(self.weights)
                    self.updatedWeights = []
            Predictions = self.predict(trainData)
            Loss = np.mean((Predictions - colorArr) ** 2)
            #print(Predictions)
            #print(Loss) Loss is too high with each x, does not change

    def predict(self, trainData):

        #Doing the feed forward part of the neural network, using training data to match with the color array
        newTrainData = np.column_stack((np.ones(len(trainData)), trainData))
        colorArr = []
        
        for x in newTrainData:
            index = self.feedForward(x)[-1][-1] #Needed to calculate loss, kept throwing operand error
            colorArr.append(index)

        colorArr = np.array(colorArr)
        return colorArr
    
    def calcReLu(self, x):
        y = np.maximum(0, x)
        return y

    def calcReluDerivative(self, x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x


class DatasetofImages(): #To extract RGB arrays from the set of sample images. TestData is the image used to
    #get the color arrays for intializing Colorization and predicting the new image from TestData
    #Training Data is a VERY SMALL data set for training data to be fed through 3 seperate neural networks: Red NN, Green NN, and Blue NN
    def __init__(self, location): #initalization
        self.location = location
        
    def extractTestData(self, testData, colorArr): #for predicting with the NNs on sample data to generate color image from BW
        result = []
        for x in testData:
            for y in x:
                result.append(y)
        redArr = []
        greenArr = []
        blueArr = []
        for x in colorArr:
            for y in x:
                redArr.append(y[0]) #extracting the 3 color arrays for RGB
                greenArr.append(y[1])
                blueArr.append(y[2])
                
        return result, redArr, greenArr, blueArr
    
    def extractTrainingData(self, location = None, sampleTrigger= False):
        if sampleTrigger == True: #switches the image collection data to sample data for predict
            self.location = location
        exts = ["jpeg","png","gif","tiff","jpg","psd","pdf"] #to stop converting images for os
        print("Opening location {}".format(self.location))
        for x, _, files in os.walk(self.location):
            if x:
                colorArr = []
                trainingData = []
                for fp in files:
                    if fp.split(".")[1] in exts: #joining all the file paths for opencv to read all the images
                        image = cv2.imread(os.path.join(x, fp))
                        image = cv2.resize(image, (50,50), interpolation = cv2.INTER_NEAREST)
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #extracting the BW image
                        red = image[:,:,2]
                        green = image[:,:,1]
                        blue = image[:,:,0]# and the BGR(CV does it backwards) for the NNs
                        (tD, cArrs) = self.dataset(gray, red, green, blue)
                        trainingData.append(tD)
                        colorArr.append(cArrs)

        return trainingData, colorArr
    #SOMETHING IS WRONG HERE, my image keeps going black
    def dataset(self, gray, red, green, blue): #generating a dataset through numpy.pad. Padding additional values helps create a more thorough(and fake)
        #dataset. Took a random value for pad width.
        greyArr = np.pad(array=gray, pad_width= 3)
        redArr = np.pad(array=red, pad_width= 3)
        greenArr = np.pad(array=green, pad_width= 3)
        blueArr = np.pad(array=blue, pad_width= 3)
        trainingData = []
        colorArr = []
        
        for i in range(0, len(greyArr)-(5)): #took a frequency value of 5 to gather arrays of inflated values and flatten them into single array of RGB i.e (R,G,B,R,G,B.....G,B)
            for j in range(0, len(greyArr)-(5)):
                trainingData.append(list(greyArr[i:i+5,j:j+5].flatten()))
                colorArr.append([redArr[i:i+5,j:j+5].flatten()[5], greenArr[i:i+5,j:j+5].flatten()[5], blueArr[i:i+5,j:j+5].flatten()[5]])
        
        return trainingData, colorArr

class Colorization(): #
    def __init__(self, redArr, blueArr, greenArr, location): #Helper Function to create the new image from the predicted arrays of RGB from the NN
        self.redArr = redArr # the predicted arrays
        self.blueArr = blueArr
        self.greenArr = greenArr
        self.size = (0, 0) #size of the image
        self.location = location
        self.open = Image.open(self.location)
        self.load = self.open.load()
        self.list = list(self.open.getdata())
        
    def createImage(self):
        dimensions = 50  #creating the image by inputing the new RGB values predicted from their respective NNs.
        count = 0
        canvas = np.zeros((dimensions, dimensions, 3), dtype=np.uint8)
        
        for i in range(dimensions):
            for j in range(dimensions):
                canvas[i,j] = (int(self.redArr[count]), int(self.greenArr[count]), int(self.blueArr[count]))
                count = count + 1
        
        image = Image.fromarray(canvas, 'RGB')
        image.save('output.jpg')
        image.show()
        
    

# Testing data and neural networks. Using sklearn for generating a dataset of mean value 0 and sd of 1
trainingImages = DatasetofImages(location= "./Data/")
trainingData, colorArr = trainingImages.extractTrainingData()
result, redArr, greenArr, blueArr = trainingImages.extractTestData(trainingData, colorArr)
scaler = StandardScaler() #I need to standardize the data to avoid high loss and poor performing CNNs, make it look like the Gaussian curve
scaler.fit(result)
NNData = scaler.transform(result) ## of epoch does not matter, keep getting high loss
redNN= NeuralNetwork(epochs = 1, hiddenNeurons = [5, 5], hiddenLayers = 2, learningRate = 0.003, batches = 25) # creating 
greeenNN= NeuralNetwork(epochs = 1, hiddenNeurons = [5, 5], hiddenLayers = 2, learningRate = 0.003, batches = 25)
blueNN= NeuralNetwork(epochs = 1, hiddenNeurons = [5, 5], hiddenLayers = 2, learningRate = 0.003, batches = 25)
print("Neural Networks successfully created. Undergoing fitting")

redNN.fit(NNData, redArr)
greeenNN.fit(NNData, greenArr)
blueNN.fit(NNData, blueArr)
    
location = "./sample/"
trainingData, colorArr = trainingImages.extractTrainingData(location, True)
greySamples, redSamples, greenSamples, blueSamples = trainingImages.extractTestData(trainingData, colorArr)
NNData = scaler.transform(greySamples)
print("undergoing predicting")    
redArr = redNN.predict(NNData)
greenArr = greeenNN.predict(NNData)
blueArr = blueNN.predict(NNData)
print("Creating image")    
createdImage = Colorization(redArr=redArr, blueArr=blueArr, greenArr=greenArr, location='./sample/sample.jpg')
createdImage.createImage()
    
