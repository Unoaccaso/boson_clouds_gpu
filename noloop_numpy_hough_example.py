import numpy
import time
from matplotlib import pyplot

#Defining the data which normally are loaded from file:
#1- matrix of f-time values, with weights, in sparse format
size = 20
offset = size*10
matrix = numpy.random.randint(2, size = size*size).reshape(size,size)
f = numpy.nonzero(matrix)[0]
times = numpy.nonzero(matrix)[1]
weights = numpy.random.rand(f.size)

#1b- load peakmap from matlab example file (see noloop_hough_example.m)
import scipy.io
peaks = scipy.io.loadmat("/home/iuri.larosa/peakmap.mat")
f = numpy.ravel(peaks["f"])
times = numpy.ravel(peaks["t"])
weights = numpy.ravel(peaks["w"])

#2- array of spindowns
nStepsfdot = 10
spindowns = numpy.arange(1,nStepsfdot+1)
print(spindowns)

#3- the size of the final matrix
nRows = spindowns.size
nColumns = numpy.int32(size+offset)

# the function to iterate
# original version
def compute_row(i):
    fShift = numpy.multiply(spindowns[i],times)
    transform = numpy.round((f-fShift)+offset).astype(numpy.int64)
    #print(transform.shape, fShift.shape, times.shape, f.shape, nColumns)
    values = numpy.bincount(transform,weights,minlength = nColumns)
    return values

# iteration performed with Python builtin function map, equivalent to for i in numpy.arange(0,nRows): 
#    image[i] = compute_row(i)
image = numpy.zeros((nRows,nColumns))
for i in numpy.arange(nRows):
    image[i] = compute_row(i)

#4- plot
#from matplotlib import pyplot
#plot = pyplot.imshow(image, aspect = 10)
#pyplot.colorbar(plot)

#5- comparison with matlab
matMap = scipy.io.loadmat('/home/iuri.larosa/oldmap.mat')['binh_df0']
#print((matMap-image)[0])
print((numpy.nonzero(matMap-image)[0]).size)
#comp_plot = pyplot.imshow(matMap-image, aspect = 10)
#pyplot.colorbar(comp_plot)
