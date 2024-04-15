import numpy as np
import pickle
import gzip
import cca_core
from CKA import linear_CKA, kernel_CKA

X = np.random.randn(100, 64)
Y = np.random.randn(100, 64)

print('Linear CKA, between X and Y: {}'.format(linear_CKA(X, Y)))
print('Linear CKA, between X and X: {}'.format(linear_CKA(X, X)))

print('RBF Kernel CKA, between X and Y: {}'.format(kernel_CKA(X, Y)))
print('RBF Kernel CKA, between X and X: {}'.format(kernel_CKA(X, X)))

# Load up second hidden layer of MNIST networks and compare
# with open("model_activations/MNIST/model_0_lay01.p", "rb") as f:
#     acts1 = pickle.load(f)
# with open("model_activations/MNIST/model_1_lay01.p", "rb") as f:
#     acts2 = pickle.load(f)
#
# print("activation shapes", acts1.shape, acts2.shape)
# print("activationT shapes", acts1.T.shape, acts2.T.shape)
# # The problem of CKA: time-consuming with large data points
# # print('Linear CKA: {}'.format(linear_CKA(acts1.T, acts2.T)))
# # print('RBF Kernel: {}'.format(kernel_CKA(acts1.T, acts2.T)))
#
# # similarity index by CCA
# results = cca_core.get_cca_similarity(acts1, acts2, epsilon=1e-10, verbose=False)
# print("Mean CCA similarity", np.mean(results["cca_coef1"]))
# Load up conv 2 activations from SVHN
with gzip.open("model_activations/SVHN/model_0_lay03.p", "rb") as f:
    acts1 = pickle.load(f)

with gzip.open("model_activations/SVHN/model_1_lay03.p", "rb") as f:
    acts2 = pickle.load(f)

print(acts1.shape, acts2.shape)


avg_acts1 = np.mean(acts1, axis=(1,2))
avg_acts2 = np.mean(acts2, axis=(1,2))
print(avg_acts1.shape, avg_acts2.shape)

# CKA
print('Linear CKA: {}'.format(linear_CKA(avg_acts1, avg_acts2)))
print('RBF Kernel CKA: {}'.format(kernel_CKA(avg_acts1, avg_acts2)))

# CCA
a_results = cca_core.get_cca_similarity(avg_acts1.T, avg_acts2.T, epsilon=1e-10, verbose=False)
print("Mean CCA similarity", np.mean(a_results["cca_coef1"]))

with gzip.open("./model_activations/SVHN/model_1_lay04.p", "rb") as f:
    pool2 = pickle.load(f)

print("shape of first conv", acts1.shape, "shape of second conv", pool2.shape)

from scipy import interpolate

num_d, h, w, _ = acts1.shape
num_c = pool2.shape[-1]
pool2_interp = np.zeros((num_d, h, w, num_c))

for d in range(num_d):
    for c in range(num_c):
        # form interpolation function
        idxs1 = np.linspace(0, pool2.shape[1],
                            pool2.shape[1],
                            endpoint=False)
        idxs2 = np.linspace(0, pool2.shape[2],
                            pool2.shape[2],
                            endpoint=False)
        arr = pool2[d, :, :, c]
        f_interp = interpolate.interp2d(idxs1, idxs2, arr)

        # creater larger arr
        large_idxs1 = np.linspace(0, pool2.shape[1],
                                  acts1.shape[1],
                                  endpoint=False)
        large_idxs2 = np.linspace(0, pool2.shape[2],
                                  acts1.shape[2],
                                  endpoint=False)

        pool2_interp[d, :, :, c] = f_interp(large_idxs1, large_idxs2)

print("new shape", pool2_interp.shape)



num_datapoints, h, w, channels = acts1.shape
f_acts1 = acts1.reshape((num_datapoints*h*w, channels))

num_datapoints, h, w, channels = pool2_interp.shape
f_pool2 = pool2_interp.reshape((num_datapoints*h*w, channels))

# CCA
f_results = cca_core.get_cca_similarity(f_acts1.T[:,::5], f_pool2.T[:,::5], epsilon=1e-10, verbose=False)
print("Mean CCA similarity", np.mean(f_results["cca_coef1"]))


# CKA
# print('Linear CKA: {}'.format(linear_CKA(f_acts1, f_pool2)))      # the shape is too large for CKA
# print('RBF Kernel CKA: {}'.format(kernel_CKA(f_acts1, f_pool2)))  # the shape is too large for CKA

