# -------------------------------------------------------------------
# This script performs a complete reconstruction workflow.
# The reconstruction algorithm used is the FBP performed on a GPU.
# -------------------------------------------------------------------
import numpy as np
import neutompy as ntp

# set pixel size in cm
pixel_size  = 0.0029

# set the last angle value of the CT scan: np.pi or 2*np.pi
last_angle = 2*np.pi

# read dataset containg projection, dark-field, flat-field images and the projection at 180 degree
proj, dark, flat, proj_180 = ntp.read_dataset()

# normalize the projections to dark-field, flat-field images and neutron dose
norm, norm_180 = ntp.normalize_proj(proj, dark, flat, proj_180=proj_180, dose_draw=True, crop_draw=True)

# rotation axis tilt correction
norm = ntp.correction_COR(norm, norm[0], norm_180)

# clean up memory
del dark; del flat; del proj; del proj_180

# remove outliers, set the optimal radius and threshold
norm = ntp.remove_outliers_stack(norm, radius=1, threshold=0.018, outliers='dark', out=norm)
norm = ntp.remove_outliers_stack(norm, radius=3, threshold=0.018, outliers='bright', out=norm)

# perform minus-log transform
norm =  ntp.log_transform(norm, out=norm)

# remove stripes from sinograms
norm = ntp.remove_stripe_stack(norm, level=4, wname='db30', sigma=1.5, out=norm)

# define the array of the angle views in radians
angles = np.linspace(0, last_angle, norm.shape[0], endpoint=False)

# FBP reconstruction with the hamming filter using GPU
print('> Reconstruction...')
rec    = ntp.reconstruct(norm, angles, 'FBP_CUDA', parameters={"FilterType":"hamming"}, pixel_size=pixel_size)

#    Implemented FilterType in ASTRA toolbox are:
#	 ``ram-lak`` (default), ``shepp-logan``, ``cosine``, ``hamming``, ``hann``, ``none``, ``tukey``,
#	 ``lanczos``, ``triangular``, ``gaussian``, ``barlett-hann``, ``blackman``, ``nuttall``,
#	 ``blackman-harris``, ``blackman-nuttall``, ``flat-top``, ``kaiser``,
#	 ``parzen``, ``projection``, ``sinogram``, ``rprojection``, ``rsinogram``.


# select the directory and the prefix file name of the reconstructed images to save.
recon_dir = ntp.save_filename_gui('', message = 'Select the folder and the prefix name for the reconstructed images...')

# write the reconstructed images to disk
ntp.write_tiff_stack(recon_dir, rec)
