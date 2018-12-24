# -------------------------------------------------------------------
# This script shows an usage example of the NN-FBP method.
# A complete dataset is reconstructed via FBP and the NN-FBP is
# trained to reconstruct some reconstructed slices using a sparse-view
# dataset. Then different slices are reconstructed via NN-FBP.
# -------------------------------------------------------------------

import numpy as np
import neutompy as ntp
import os

# set pixel size in cm
pixel_size  = 0.0029

hqrec_folder = 'hqrecs/'  # folder to save high quality reconstuction
nnfbp_rec_folder = 'recon-nnfbp/' # output folder of the nn-fbp reconstruction
conf = {}
conf['hidden_nodes']  = 3 # number of hidden nodes
conf['hqrecfiles']    = hqrec_folder + 'sample*.tiff' # high-quality reconstruction files
conf['traindir']      = 'trainfiles/'  # folder where training files are stored
conf['npick']         = 10000          # number of random pixels to pick per slice
conf['filter_file']   = 'filters.mat'  # file to store trained filters

# set the last angle value of the CT scan: np.pi or 2*np.pi
last_angle = 2*np.pi

# read dataset containing projection, dark-field, flat-field images and the projection at 180 degree
proj, dark, flat, proj_180 = ntp.read_dataset()

# normalize the projections to dark-field, flat-field images and neutron dose
norm, norm_180 = ntp.normalize_proj(proj, dark, flat, proj_180=proj_180, dose_draw=True, crop_draw=True, log=True)

# rotation axis tilt correction
norm = ntp.correction_COR(norm, norm[0], norm_180)

# define the array of the angle views in radians
angles = np.linspace(0, last_angle, norm.shape[0], endpoint=False)

# high-quality reconstruction
train_slice_start = 100
train_slice_end  = 120
rec = ntp.reconstruct(norm[:,train_slice_start:train_slice_end+1, :], angles, 'FBP_CUDA', parameters={"FilterType":"hamming"}, pixel_size=pixel_size)

# write the high-quality reconstructed images to disk
try:
	os.mkdir(hqrec_folder)
except OSError:
	pass
ntp.write_tiff_stack(hqrec_folder + 'sample', rec)

# NN-FBP training
skip  = 3  # reduction factor of the full dataset to obtain the sparse-view dataset
norm_train = norm[::skip,train_slice_start:train_slice_end+1, :]
ntp.reconstruct(norm_train, angles[::skip], 'NN-FBP-train', parameters=conf)

# NN-FBP reconstruction of noisy projections
test_slice_start = 180
test_slice_end   = 200
norm_test = norm[::skip,test_slice_start:test_slice_end+1, :]
rec_nnfbp = ntp.reconstruct(norm_test, angles[::skip], 'NN-FBP', parameters=conf)

# write NN-FBP reconstructed images
try:
	os.mkdir(nnfbp_rec_folder)
except OSError:
	pass
ntp.write_tiff_stack(nnfbp_rec_folder + 'sample', rec_nnfbp)
