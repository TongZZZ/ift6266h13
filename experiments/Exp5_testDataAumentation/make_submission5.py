import sys
import numpy as np
from PIL import Image

def usage():
    print """usage: python make_submission.py model.pkl submission.csv
Where model.pkl contains a trained pylearn2.models.mlp.MLP object.
The script will make submission.csv, which you may then upload to the
kaggle site."""

def genTransformations(case):
    
    img = case.reshape(48, 48)
    transformations = None
    
    for reflexion in [True, False]:
        
        img2 = np.copy(img)
        if reflexion:
            img2 = img2[:,::-1]
        
        for translation in [(-1, -1), (-1, 1), (1, -1), (1, 1), (0, 0)]:
        
            h_translation = translation[0]
            v_translation = translation[1]
        
            img3 = np.copy(img2)
            
            # Perform horizontal translation 
            if h_translation < 0:
                temp = img3[:,-h_translation:]
                img3[:,:h_translation] = temp
                img3[:,h_translation:] = 0
            elif h_translation > 0:
                temp = img3[:,:-h_translation]
                img3[:,h_translation:] = temp
                img3[:,:h_translation] = 0
                             
            # Perform vertical translation 
            if v_translation < 0:
                temp = img3[-v_translation:,:]
                img3[:v_translation,:] = temp
                img3[v_translation:,:] = 0
            elif v_translation > 0:
                temp = img3[:-v_translation,:]
                img3[v_translation:,:] = temp
                img3[:v_translation,:] = 0  
            
            for rotation in [0]:
            
                img4 = np.array(Image.fromarray(img3).rotate(rotation)).reshape(2304)
                
                if transformations == None:
                    transformations = img4
                else:
                    transformations = np.vstack((transformations, img4))
    
    return transformations.reshape(transformations.shape[0], 2304)


if len(sys.argv) != 3:
    usage()
    print "(You used the wrong # of arguments)"
    quit(-1)

_, model_path, out_path = sys.argv

import os
if os.path.exists(out_path):
    usage()
    print out_path+" already exists, and I don't want to overwrite anything just to be safe."
    quit(-1)

from pylearn2.utils import serial
try:
    model = serial.load(model_path)
except Exception, e:
    usage()
    print model_path + "doesn't seem to be a valid model path, I got this error when trying to load it: "
    print e

from pylearn2.config import yaml_parse

dataset = yaml_parse.load(model.dataset_yaml_src)
dataset = dataset.get_test_set()


# Sneaky addition
# To get advantage of a model's invariance to translation
# rotation, scaling and/or reflexion, we create many variations of each test
# case. The model will then be run on all of the variations of all the test
# cases. The prediction will be veraged for each test case to get the final
# prediction for that test case. Ties are broken by looking at the model's
# prediction on the original image.
test_points = dataset.X
dataset.X = np.vstack([genTransformations(case) for case in test_points])
nbTransformByTestCase = dataset.X.shape[0] / test_points.shape[0]
# End of sneaky addition


# use smallish batches to avoid running out of memory
batch_size = 100
model.set_batch_size(batch_size)
# dataset must be multiple of batch size of some batches will have
# different sizes. theano convolution requires a hard-coded batch size
m = dataset.X.shape[0]
extra = batch_size - m % batch_size
assert (m + extra) % batch_size == 0
if extra > 0:
    dataset.X = np.concatenate((dataset.X, np.zeros((extra, dataset.X.shape[1]),
    dtype=dataset.X.dtype)), axis=0)
assert dataset.X.shape[0] % batch_size == 0

X = model.get_input_space().make_batch_theano()
Y = model.fprop(X)

from theano import tensor as T

y = Y

from theano import function

f = function([X], y)


y = []

for i in xrange(dataset.X.shape[0] / batch_size):
    x_arg = dataset.X[i*batch_size:(i+1)*batch_size,:]
    if X.ndim > 2:
        x_arg = dataset.get_topological_view(x_arg)
    y.append(f(x_arg.astype(X.dtype)))

y = np.concatenate(y)


assert y.shape[0] == dataset.X.shape[0]
# discard any zero-padding that was used to give the batches uniform size
y = y[:m]


# Sneaky addition
# Average predictions over all variations of a same test case
average_y = []
for i in xrange(test_points.shape[0]):
    means = y[i*nbTransformByTestCase:(i+1)*nbTransformByTestCase].mean(0)
    average_y.append(means.argmax())
y = np.array(average_y)
# End of sneaky addition


out = open(out_path, 'w')
for i in xrange(y.shape[0]):
    out.write('%d\n' % y[i])
out.close()


