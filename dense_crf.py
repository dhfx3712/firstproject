import numpy as np
import pydensecrf.densecrf as dcrf
import torch.nn.functional as F
import torch

np.random.seed(10)
d = dcrf.DenseCRF2D(640, 480, 5)  # width, height, nlabels
print (d)
U = np.random.randint(5,size=(5,480,640))
print (f'U_max : {np.argmax(U,axis=0)},U_shape :{np.shape(np.argmax(U,axis=0))}')
x= torch.Tensor(U)
# print (np.shape(U))
U = F.softmax(x)
# print (U)
U = U.numpy()


im = np.random.randint(255,size=(480,640,3))

# im = im.astype(np.float32)
im = im.astype(np.uint8)
# U = np.array(data)     # Get the unary in some way.
print(U.shape)        # -> (5, 480, 640)
print(U.dtype)        # -> dtype('float32')
U = U.reshape((5,-1)) # Needs to be flat.
d.setUnaryEnergy(U)

# Or alternatively: d.setUnary(ConstUnary(U))

d.addPairwiseGaussian(sxy=3, compat=3)
d.addPairwiseBilateral(sxy=80, srgb=13, rgbim=im, compat=10)
Q = d.inference(5)

# print (np.shape(Q))
proba = np.array(Q)
print (proba,np.shape(proba))

Q = np.argmax(proba, axis=0).reshape((480,640))
print (np.shape(Q),Q)

