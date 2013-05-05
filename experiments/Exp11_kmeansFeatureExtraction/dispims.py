import numpy
import pylab

# This file has been obtained from http://www.iro.umontreal.ca/~memisevr/teaching/mlvis2013/
# The original author is Roland Memisevic

def dispims(M, height, width, border=0, bordercolor=0.0, layout=None, **kwargs):
    """ Display a stack of vectorized matrices (colunmwise). 

    """
    numimages = M.shape[1]
    if layout is None:
        n0 = numpy.int(numpy.ceil(numpy.sqrt(numimages)))
        n1 = numpy.int(numpy.ceil(numpy.sqrt(numimages)))
    else:
        n0, n1 = layout
    im = bordercolor * numpy.ones(((height+border)*n0+border,(width+border)*n1+border),dtype='<f8')
    for i in range(n0):
        for j in range(n1):
            if i*n1+j < M.shape[1]:
                im[i*(height+border)+border:(i+1)*(height+border)+border,
                   j*(width+border)+border :(j+1)*(width+border)+border] = numpy.vstack((
                            numpy.hstack((numpy.reshape(M[:,i*n1+j],(height, width)),
                                   bordercolor*numpy.ones((height,border),dtype=float))),
                            bordercolor*numpy.ones((border,width+border),dtype=float)
                            ))
    pylab.imshow(im, cmap=pylab.cm.gray, interpolation='nearest', **kwargs)
    pylab.show()
    
def dispims_color(M, border=0, bordercolor=[0.0, 0.0, 0.0], savePath=None, *imshow_args, **imshow_keyargs):
    """ Display an array of rgb images. 

    The input array is assumed to have the shape numimages x numpixelsY x numpixelsX x 3
    """
    bordercolor = numpy.array(bordercolor)[None, None, :]
    numimages = len(M)
    M = M.copy()
    for i in range(M.shape[0]):
        M[i] -= M[i].flatten().min()
        M[i] /= M[i].flatten().max()
    height, width, three = M[0].shape
    assert three == 3
    
    n0 = numpy.int(numpy.ceil(numpy.sqrt(numimages)))
    n1 = numpy.int(numpy.ceil(numpy.sqrt(numimages)))
    im = numpy.array(bordercolor)*numpy.ones(
                             ((height+border)*n1+border,(width+border)*n0+border, 1),dtype='<f8')
    for i in range(n0):
        for j in range(n1):
            if i*n1+j < numimages:
                im[j*(height+border)+border:(j+1)*(height+border)+border,
                   i*(width+border)+border:(i+1)*(width+border)+border,:] = numpy.concatenate((
                  numpy.concatenate((M[i*n1+j,:,:,:],
                         bordercolor*numpy.ones((height,border,3),dtype=float)), 1),
                  bordercolor*numpy.ones((border,width+border,3),dtype=float)
                  ), 0)
    imshow_keyargs["interpolation"]="nearest"
    pylab.imshow(im, *imshow_args, **imshow_keyargs)
    
    if savePath == None:
        pylab.show()
    else:
        pylab.savefig(savePath)
