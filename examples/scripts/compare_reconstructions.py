# -------------------------------------------------------------------
# This script performs a FBP, SIRT and CGLS reconstruction of a
# phantom sample. The reconstructions are compared using several
# image quality indexis.
# -------------------------------------------------------------------

import numpy as np
import neutompy as ntp
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

# pixel size in cm
pixel_size = 0.0029

# factor to reduce the number of projections in the sinogram
skip_theta = 3

# image filename
fname = 'sinogram.tiff'

# read the sinogram
sino_hq = ntp.read_image(fname)
na     = sino_hq.shape[0]

# reduce number of projections in the sinogram
sino   = sino_hq[::skip_theta]

# angles views in radians
angles = np.linspace(0, 2*np.pi, na, False)[::skip_theta]

# ground truth reconstruction
true = ntp.reconstruct(sino_hq, np.linspace(0, np.pi*2, sino_hq.shape[0], endpoint=False), 'SIRT_CUDA',
						{'iterations':200}, pixel_size=pixel_size)
# fbp reconstruction
fbp  = ntp.reconstruct(sino, angles, 'FBP_CUDA', pixel_size=pixel_size)
# sirt reconstruction
sirt = ntp.reconstruct(sino, angles, 'SIRT_CUDA', parameters={'iterations':200}, pixel_size=pixel_size)
# cgls reconstruction
cgls = ntp.reconstruct(sino, angles, 'CGLS_CUDA', parameters={'iterations':10}, pixel_size=pixel_size)

# define a list of reconstructed images
rec_list = [fbp, sirt, cgls]
rec_name = ['FBP', 'SIRT', 'CGLS']

# roi coordinates
rmin = 0
rmax = None
cmin = 0
cmax = 880

# set the x-axis range of the histograms
xmin = 0.0
xmax = 0.7

# counts range of the histograms
ymin = 10
ymax = 2e4

nsquare = 3
nbins = 300

[binning, width] = np.linspace(xmin, xmax, nbins, retstep=True)

nsubplot = len(rec_list)

plt.rc('font', family='serif', serif='Times', size=11)
plt.rc('text', usetex=False)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
plt.rc('axes', labelsize=12)


fig = plt.figure(figsize=(nsquare*nsubplot,nsquare+1+0.5))
fig.subplots_adjust(hspace=0, wspace=0.5, top=0.8)
gs = GridSpec(nsquare+1,nsquare*nsubplot) # 2 rows, 3 columns

for i in range(0, nsubplot):

	ax1=fig.add_subplot(gs[0:nsquare,i*nsquare:(i+1)*nsquare])
	# quality metrics evaluation
	img = rec_list[i]
	im = ax1.imshow(img[rmin:rmax, cmin:cmax], vmin=xmin, vmax=xmax, cmap='gray')
	ssim  = ntp.SSIM(img, true)
	nrmse = ntp.NRMSE(img, true)
	cnr   = ntp.CNR(img, froi_signal='signal.roi', froi_background='background.roi')

	title = rec_name[i]
	plt.title(title +  '\n SSIM = '+ "{:.2f}".format(ssim) + ', CNR = ' "{:.1f}".format(cnr) + ',\n NRMSE = '+ "{:.2f}".format(nrmse))
	plt.xticks([])
	plt.yticks([])

	ax2=fig.add_subplot(gs[nsquare,i*nsquare:(i+1)*nsquare]) # Second row, span all columns
	# generate histogram of the gray values inside a circular mask
	mask = ntp.get_circular_mask(img.shape[0], img.shape[1], radius=370, center=(img.shape[0]//2, img.shape[0]//2 -30))
	cc, edge = np.histogram(img[mask], bins=binning)
	ax2.bar(edge[:-1]+width*0.5, cc, width, color='C3', edgecolor='C3', log=True)
	plt.xlim([0, xmax])
	plt.ylim([ymin, ymax])
	plt.yticks([1e2, 1e3, 1e4], ['']*3)
	if i == 0:
		plt.yticks([1e2, 1e3, 1e4], ['$10^2$', '$10^3$', '$10^4$'])


plt.show()
