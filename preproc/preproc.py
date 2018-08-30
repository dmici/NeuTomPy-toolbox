import numpy as np
from numpy import sin, cos
import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
from skimage.transform import rotate
from matplotlib.offsetbox import AnchoredText
import numexpr as ne
from read_roi import read_roi_file
import cv2
import SimpleITK as sitk
from neutompy.image.image import get_rect_coordinates_from_roi
from neutompy.misc.uitools import get_screen_resolution
from tqdm import tqdm
import sys

__author__  = "Davide Micieli"
__version__ = "0.1.0"
__all__     = ['draw_ROI',
			   'normalize_proj',
			   'log_transform',
			   'correction_COR',
			   'remove_outliers',
			   'find_COR',
			   'remove_outliers_stack'
			  ]

def draw_ROI(img, title, ratio=0.85):
	"""
	This function allow to select interactively a rectangular region of interest over an image.
	The function returns the ROI coordinates.

	Parameters
	----------

	img : 2d array
		The image on which the dose roi is drawn.

	title : str
		String defining the title of the window shown.

	ratio : float, optional
		The filling ratio of the windows respect to the screen resolution.
    It must be a number between 0 and 1. The default value is 0.8.

	Returns
	-------
	rowmin : int
		The minimum row coordinate.

	rowmax : int
		The maximum row coordinate.

	colmin : int
		The minimum column coordinate.

	colmax : int
		The maximum column coordinate.

	"""
	if not (0 < ratio <= 1):
		raise ValueError('The variable ratio must be between 0 and 1.')

	# window size settings
	(width, height) = get_screen_resolution()
	scale_width = width / img.shape[1]
	scale_height = height / img.shape[0]
	scale = min(scale_width, scale_height)*ratio
	window_width = int(img.shape[1] * scale)
	window_height = int(img.shape[0] * scale)

	img = img.astype(np.float32)
	mu = np.nanmedian(img.ravel())
	finite_vals = np.nonzero(np.isfinite(img))
	s  = img[finite_vals].std()
	img = img/(mu+1*s)  # normalization can be improved
	imgshow = np.clip(img, 0, 1.0)

	condition = True
	while condition:
		cv2.namedWindow(title, cv2.WINDOW_NORMAL)
		cv2.resizeWindow(title,  window_width, window_height)
		r = cv2.selectROI(title, imgshow, showCrosshair = False, fromCenter = False)

		if (r == (0,0,0,0)):
			condition = True
			print('> ROI cancelled')
		else:
			condition = False
			#print('> ROI selected ')
			rowmin, rowmax, colmin, colmax = r[1], r[1] + r[3], r[0], r[0] + r[2]
			print( 'ROI selected: ymin =', rowmin, ', ymax =', rowmax, ', xmin =', colmin, ', xmax =', colmax)
			cv2.destroyWindow(title)

	return rowmin, rowmax, colmin, colmax




def _normalize (proj, dark, flat, mode='mean', min_denom=1.0e-6, out=None):

	if(min_denom<=0.0):
		raise ValueError('The parameter min_ratio must be positive.')

	if (mode == 'mean'):
		func = np.mean
	elif(mode == 'median'):
		func = np.median
	else:
		raise ValueError('Not valid function for projecting flat and dark images along z axis. Set mean or median')

	mean_flat  = func(flat, axis=0).astype(np.float32)
	mean_dark  = func(dark, axis=0).astype(np.float32)

	min_denom = np.float32(min_denom)

	# numerator
	out = np.zeros(proj.shape, dtype=np.float32)
	out = ne.evaluate('proj-mean_dark', out = out)

	# denominator
	den = ne.evaluate('mean_flat-mean_dark')
	den = ne.evaluate('where(den<min_denom, min_denom, den)', out = den)

	ne.evaluate('out/den', out=out, truediv=True)

	return out


def normalize_proj(proj, dark, flat,  proj_180=None, out=None,
						 dose_file='', dose_coor=(), dose_draw=True,
						 crop_file='', crop_coor=(), crop_draw=True,
						 min_denom=1.0e-6,  min_ratio=1e-6, max_ratio=10.0,
						 mode='mean', log=False,  sino_order=False, show_opt='mean'):


	if(min_ratio<=0.0):
		raise ValueError('The parameter min_ratio must be positive.')

	if(max_ratio<=0.0):
		raise ValueError('The parameter min_ratio must be positive.')

	if(max_ratio<=min_ratio):
		raise ValueError('Invalid values assigned to max_ratio and min_ratio.')

	if not (proj.ndim == 3 and dark.ndim == 3 and flat.ndim == 3):
		raise ValueError('All images stack must have three dimesions.')


	if type(proj_180) is np.ndarray:
		if (proj_180.ndim !=2):
			raise ValueError('Invalid array dimensions. The projection at 180 must be a 2D array.')
		AddProj180 = True


	if proj_180 is None:
		AddProj180 = False

	# only one DOSE roi selection check
	doseON = 0
	if (dose_file):
		doseON = doseON + 1
	if (dose_coor):
		doseON = doseON + 1
	if (dose_draw):
		doseON = doseON + 1
	if(doseON>=2):
		raise ValueError('Only one selection method of the dose ROI is allowed.')

	# only one CROP roi selection check
	cropON = 0
	if (crop_file):
		cropON = cropON + 1
	if (crop_coor):
		cropON = cropON + 1
	if (crop_draw):
		cropON = cropON + 1
	if(cropON>=2):
		raise ValueError('Only one selection method of the ROI to crop is allowed.')

	if (show_opt=='mean'):
		func_show = np.mean
	elif (show_opt=='std'):
		func_show = np.std
	else:
		raise ValueError('Not valid value of the variable show_opt. Choose "mean" or "std" to show the mean or the standard deviation computed pixel-wise over all projections.')

	# get the ROI to CROP
	if(cropON):
		# read ImageJ ROI
		if(crop_file):
			rmin_c, rmax_c, cmin_c, cmax_c = get_rect_coordinates_from_roi(crop_file)

		# read coordinate from tuple
		if (crop_coor):
			rmin_c, rmax_c, cmin_c, cmax_c = crop_coor

		# draw the roi over the image
		if (crop_draw):
			show_proj = func_show(proj, axis=0, dtype=np.float32)
			rmin_c, rmax_c, cmin_c, cmax_c = draw_ROI(show_proj, 'Select the region to use for the reconstruction...' )

		proj_c = proj[:, rmin_c:rmax_c, cmin_c:cmax_c]
		dark_c = dark[:, rmin_c:rmax_c, cmin_c:cmax_c]
		flat_c = flat[:, rmin_c:rmax_c, cmin_c:cmax_c]

		crop_roi = rmin_c, rmax_c, cmin_c, cmax_c

	else:
		proj_c = proj
		dark_c = dark
		flat_c = flat
		crop_roi = None

	# get the ROI to DOSE
	if(doseON):
		# read ImageJ ROI
		if(dose_file):
			rmin_d, rmax_d, cmin_d, cmax_d = get_rect_coordinates_from_roi(dose_file)

		# read coordinate from tuple
		if (dose_coor):
			rmin_d, rmax_d, cmin_d, cmax_d  = dose_coor

		# draw the roi over the image
		if (dose_draw):
			if not crop_draw: # to prevent double computation of std
				show_proj = func_show(proj, axis=0, dtype=np.float32)
			rmin_d, rmax_d, cmin_d, cmax_d  = draw_ROI(show_proj, 'Select a region free of the sample...' )

		ds_roi = rmin_d, rmax_d, cmin_d, cmax_d


		if(mode=='mean'):
			func = np.mean
		elif(mode=='median'):
			func = np.median
		else:
			raise ValueError('Not valid function for projecting flat and dark images along z axis.Set mean or media.')

		mean_flat  = func(flat[:, rmin_d:rmax_d, cmin_d:cmax_d], axis=0).astype(np.float32)
		mean_dark  = func(dark[:, rmin_d:rmax_d, cmin_d:cmax_d], axis=0).astype(np.float32)

		min_denom   = np.float32(min_denom)
		min_ratio   = np.float32(min_ratio)
		max_ratio   = np.float32(max_ratio)

		proj_d  = proj[:, rmin_d:rmax_d, cmin_d:cmax_d]
		num_d   = ne.evaluate('proj_d - mean_dark')
		den_d   = ne.evaluate('mean_flat - mean_dark')

		# (flat - dark) dose
		D0 = np.median(den_d)

		# projection dose computed in a roi free-of-sample
		D  = np.median(num_d, axis=(1,2))

		# correction factor
		if(proj.shape[0]==1):
			k = D0/D
		else:
			k = D0/(D.reshape(proj.shape[0], 1))

	# end dose computation
	else:
		ds_roi = None

	out = _normalize(proj_c, dark_c, flat_c, mode, min_denom,  out=out)

	if(doseON):
		if(proj.shape[0]==1):
			out = k*out
		else:
			nz, ny, nx = out.shape
			out = (k*out.reshape(nz, nx*ny)).reshape(nz, ny, nx)


	min_ratio = np.float32(min_ratio)
	max_ratio = np.float32(max_ratio)

	out = ne.evaluate('where(out>max_ratio, max_ratio, out)', out = out)
	out = ne.evaluate('where(out<min_ratio, min_ratio, out)', out = out)


	if (log):
		out = ne.evaluate('-log(out)', out=out)


	if (sino_order):
		out = out.swapaxes(0,1)

	if(AddProj180):
		proj_180 = np.expand_dims(proj_180, axis=0)

		out_180  = normalize_proj(proj_180, dark, flat,  proj_180=None, out=None,
						 dose_file='', dose_coor = ds_roi, dose_draw=False,
						 crop_file='', crop_coor = crop_roi, crop_draw=False,
						 min_denom=min_denom,  min_ratio=min_ratio, max_ratio=max_ratio,
						 mode=mode, log=log,  sino_order=sino_order)

		return out, out_180[0]

	else:
		return out



def log_transform(norm_proj, out=None):
	"""
	This function computes the minus log of an array. In trasmission CT the input array must be the normalized dataset after flat-fielding correction.

	Parameters
    ----------
    norm_proj : ndarray
        3D stack of projections.

    out : ndarray, optional
        Output array. If same as norm_proj, computation will be done in-place.

    Returns
    -------
    out : ndarray
        Minus-log of the input array.

	"""

	out = ne.evaluate('-log(norm_proj)', out=out)

	return out


def resample(image, transform, interpolator=sitk.sitkLinear):

    dims = image.GetSize()
    reference_image = sitk.Image(dims[0], dims[1], sitk.sitkFloat32)
    default_value = 1.0

    return sitk.Resample(image, reference_image, transform, interpolator, default_value)


def rotate_2(img, theta, interpolator=sitk.sitkLinear):

	s = sitk.GetImageFromArray(img)
	aff = sitk.AffineTransform(2)

	com = np.array(list(img.shape))*0.5

	th   = -np.deg2rad(theta)
	matr = np.array([[cos(th), sin(th)] , [-sin(th), cos(th)]])


	center = np.array(img.shape)*0.5 - 0.5
	aff.SetCenter(center[::-1])
	#shift = np.array( com - center)
	#aff.SetTranslation ( -np.dot(matr,shift))

	aff.SetMatrix(matr.flatten())

	out= resample(s, aff)
	outimg = sitk.GetArrayFromImage(out)

	return outimg


def rotate_sitk(img, theta, interpolator=sitk.sitkLinear):

	th = -np.deg2rad(theta)
	l1,l2 = img.shape
	imgpad_size = np.array([ int(round(abs(l1*cos(th)) + abs(l2*sin(th)))), int(round(abs(l2*cos(th)) + abs(l1*sin(th)))) ])


	before =  (imgpad_size - np.array(img.shape))//2
	after  =  imgpad_size -  np.array(img.shape) -before

	before = tuple(before)
	after  = tuple(after)

	imgpad = np.pad(img, pad_width=((before[0], after[0]),  (before[1], after[1])), mode='edge')


	out = rotate_2(imgpad, theta, interpolator)

	return out[ before[0]:-after[0],before[1]:-after[1] ]



def find_COR(proj_0, proj_180, nroi=None, ref_proj=None, ystep=5, ShowResults=True):



	if (ref_proj is None):
		ref_proj = proj_0

	proj_0   = proj_0.astype(np.float32)
	proj_180 = proj_180.astype(np.float32)

	# number of pixels in a row
	nd = proj_0.shape[1]
	# number of slices
	nz = proj_0.shape[0]

	# array containing that contains the z points within the rois selected by the user
	slices = np.array([], dtype=np.int32)

	print('> Finding the rotation axis position...')
	# set ROI number
	if not nroi:
		print('It is necessary to select one or multiple regions where the sample is present.\nThen you must draw the different regions vertically starting from top to bottom.')
		while True:
			nroi = input('> Insert the number of regions to select: ')
			if(nroi.isdigit() and int(nroi)>0):
				nroi = int(nroi)
				break
			else:
				print('Not valid input.')

	tmin = - nd//2
	tmax =   nd - nd//2


	# ROI selection
	for i in range(0, nroi):
		# print number of rois
		print('> Select ROI ' + str(i+1))
		title = 'Select region n. : ' + str(i+1)
		ymin, ymax, _, _ = draw_ROI(ref_proj, title)

		aus = np.arange(ymin, ymax +1, ystep)
		slices = np.concatenate((slices, aus), axis=0)

	shift = np.zeros(slices.size)
	proj_flip = proj_180[:, ::-1]

	# loop over different rows
	for z, slc in enumerate(slices):

		minimum = 1e7
		index_min = 0

		# loop over the shift
		for t in range(tmin, tmax + 1):

			posz = np.round(slc).astype(np.int32)
			rmse = np.square( (np.roll(proj_0[posz], t, axis = 0) - proj_flip[posz]) ).sum() / nd

			if(rmse <= minimum):
				minimum = rmse
				index_min = t

		shift[z] = index_min

	# perform linear fit
	par = np.polyfit(slices, shift, deg=1)
	m = par[0]
	q = par[1]

	# compute the tilt angle
	theta = np.arctan(0.5*m)   # in radians
	theta = np.rad2deg(theta)

	# compute the shift
	offset       = (np.round(m*nz*0.5 + q)).astype(np.int32)*0.5
	middle_shift = (np.round(m*nz*0.5 + q)).astype(np.int32)//2


	print("COR Found!")
	print("offset =", offset, "   tilt angle =", theta, "°"  )

	p0_r = np.zeros(proj_0.shape, dtype=np.float32)
	p90_r = np.zeros(proj_0.shape, dtype=np.float32)


	# plot difference proj_0 - proj_180_flipped
	plt.figure('Analysis of the rotation axis position', figsize=(14,5), dpi=96)
	plt.subplots_adjust(wspace=0.5)
	ax1 = plt.subplot(1,2,1)

	diff = proj_0 - proj_flip
	mu = np.median(diff)
	s  = diff.std()

	plt.imshow(diff, cmap='gray', vmin=mu-s, vmax=mu+s)

	info_cor = 'offset = '  + "{:.2f}".format(offset) + '\n       θ = ' + "{:.3f}".format(theta)
	anchored_text1 = AnchoredText(info_cor, loc=2)
	ax1.add_artist(anchored_text1)

	plt.title('$P_0 - P^{flipped}_{\pi}$')
	plt.colorbar(fraction=0.046, pad=0.04)


	zaxis = np.arange(0, nz)
	# plot fit
	plt.plot( nd*0.5 - 0.5*m*zaxis - 0.5*q, zaxis,'b-')
	# plot data
	plt.plot( nd*0.5 - 0.5*shift, slices, 'r.', markersize=3)
	# draw vertical axis
	plt.plot([0.5*nd, 0.5*nd], [0, nz-1], 'k--' )

	# show results of the fit
	ax2 = plt.subplot(1,2,2)
	info_fit = 'shift = '  + "{:.3f}".format(m) + '*y + ' + "{:.3f}".format(q)
	anchored_text2 = AnchoredText(info_fit, loc=9)
	plt.plot(zaxis, m*zaxis + q, 'b-', label = 'fit')
	plt.plot(slices, shift, 'r.', label='data')
	plt.xlabel('$y$')
	plt.ylabel('shift')
	plt.title('Fit result')
	ax2.add_artist(anchored_text2)

	plt.legend()


	#~ p0_r = np.roll(rotate(proj_0, theta, preserve_range=True,order=0, mode='edge'),    middle_shift , axis=1)
	#~ p90_r = np.roll(rotate(proj_180, theta, preserve_range=True, order=0, mode='edge'),  middle_shift, axis=1)
	p0_r = np.roll(rotate_sitk(proj_0, theta, interpolator=sitk.sitkLinear),    middle_shift , axis=1)
	p90_r = np.roll(rotate_sitk(proj_180, theta, interpolator=sitk.sitkLinear),  middle_shift, axis=1)


	# FIGURE with difference image and histogram
	plt.figure('Results of the rotation axis correction', figsize=(14,5), dpi=96)

	plt.subplot(1,2,1)
	plt.subplots_adjust(wspace=0.5)

	diff2 = p0_r - p90_r[:,::-1]
	mu = np.median(diff2)
	s  = diff2.std()
	plt.imshow(diff2 , cmap='gray', vmin=mu-s, vmax=mu+s)
	plt.title('$P_0 - P^{flipped}_{\pi}$ with correction')
	plt.colorbar(fraction=0.046, pad=0.04)



	# histogram of squares of residuals
	ax3 = plt.subplot(1,2,2)
	nbins = 1000
	row_marg = int(0.1*diff2.shape[0])
	col_marg = int(0.1*diff2.shape[1])
	absdif = np.abs(diff2)[row_marg:-row_marg, col_marg:-col_marg]
	[binning, width] = np.linspace(absdif.min(), absdif.max(), nbins, retstep=True)
	cc, edge = np.histogram(absdif, bins=binning)
	plt.bar(edge[:-1]+width*0.5, cc, width, color='C3', edgecolor='k', log=True)
	plt.gca().set_xscale("log")
	plt.gca().set_yscale("log")
	plt.xlim([0.01, absdif.max()])
	plt.xlabel('Residuals')
	plt.ylabel('Entries')
	plt.title('Histogram of residuals')

	# write text about residuals
	res =  np.abs(diff2).mean()
	info_res = '$||P_0 - P^{flipped}_{\pi}||_1 / N_{pixel}$ = ' + "{:.4f}".format(res)
	anchored_text3 = AnchoredText(info_res, loc=1)
	ax3.add_artist(anchored_text3)

	print("average of residuals  = ", res)


	if(ShowResults):
		plt.show(block=False)

	return middle_shift, theta


def correction_COR(norm_proj, proj_0, proj_180, show_opt='mean', shift=None, theta=None, nroi=None, ystep=5):
	"""
	This function computes the 
	specificare che l'array deve essere uno stack di proiezioni e non uno stack di sinogrammi
	"""

	# compute COR axis interactively, if shift and theta are not known
	if (shift==None and theta==None):

		if (show_opt == 'mean'):
			proj2show = norm_proj.mean(axis=0, dtype=np.float32)
		elif (show_opt == 'std'):
			proj2show = norm_proj.std(axis=0, dtype=np.float32)
		elif (show_opt == 'zero'):
			proj2show = proj_0
		elif (show_opt == 'pi'):
			proj2show = proj_180
		else:
			raise ValueError('Flag show_opt not valid. Allowed values are `mean`, `std`, `zero` or `pi`.')

		condition = True

		while condition:

			shift, theta = find_COR(proj_0, proj_180, nroi=nroi, ref_proj=proj2show, ystep=ystep, ShowResults=True)

			while True:
					ans = input('> COR found. Do you want to correct all projections? \n[Y] Yes, correct them.  \n[N] No, find again the COR. \n[C] Cancel and abort the script. \nType your answer and press [Enter] :')
					if(ans=='Y' or ans=='y'):
						plt.close('all')
						condition = False
						break
					elif(ans=='N' or ans=='n'):
						plt.close('all')
						break
					elif(ans=='C' or ans=='c'):
						plt.close('all')
						print('> Script aborted.')
						sys.exit()
					else:
						print('Input not valid.')

	# end if

	# correct all projections
	print('> Correcting center of rotation misalignment...')
	for s in tqdm(range(0, norm_proj.shape[0]), unit=' images'):
		# norm_proj[s,:,:] = np.roll(rotate(norm_proj[s,:,:], theta, preserve_range=True, order=1, mode='edge'), shift, axis=1)
		norm_proj[s,:,:] = np.roll(rotate_sitk(norm_proj[s,:,:], theta, interpolator=sitk.sitkLinear), shift, axis=1)

	return norm_proj


def median_filter(img, size):

	simg = sitk.GetImageFromArray(img)
	mf = sitk.MedianImageFilter()
	mf.SetRadius(size)
	fimg = sitk.GetArrayFromImage(mf.Execute(simg))

	return fimg

def convolution_2D (img, kernel, boundary='ZERO_PAD', out_mode='SAME'):
	"""
	Perform a convolution with a custom kernel.

	Parameters
	----------
	img : 2d array
		Set the variable type

	kernel : 2d array
		Set the variable array

	boundary: str

	out_mode: str

	Returns
	-------
	out : 2d array
		The filtered image. The type of the array is inferred from img and kernel type.

	"""
	simg    = sitk.GetImageFromArray(img)
	skernel = sitk.GetImageFromArray(kernel)

	conv = sitk.ConvolutionImageFilter()
	conv.SetBoundaryCondition(eval('conv.' + boundary))
	conv.SetOutputRegionMode(eval('conv.' + out_mode))

	filt = sitk.GetArrayFromImage(conv.Execute(simg, skernel))

	return filt

def std_map(img, size):

	if (type(size) is not int or size==0):
		raise ValueError('Kernel size must be a positive integer.')

	vartype = img.dtype # get the array type
	img     = img.astype(vartype)
	img2    = img**2
	ones    = np.ones(img.shape, dtype=vartype)
	kernel = np.ones((2*size+1, 2*size+1),  dtype=vartype)

	s  = convolution_2D(img, kernel)
	s2 = convolution_2D(img2, kernel)
	ns = convolution_2D(ones, kernel)

	return np.sqrt((s2 - s**2 / ns) / (ns-1))


def remove_outliers(img, size, threshold, outliers='bright', k=1.0, out=None):

	if not (threshold=='local' or type(threshold) in [float, int]):
		raise ValueError('The threshold must be a float variable or the string `local`.')

	if(threshold=='local'):
		threshold = std_map(img, size)

	median = median_filter(img, size)

	if(outliers=='bright'):
		diff    = img - median
		spot    = diff > threshold * k
	elif(outliers=='dark'):
		diff    =  median - img
		spot    = diff   > threshold * k

	# convert type
	cut = np.array(threshold*k, dtype = img.dtype)

	out = ne.evaluate('where(diff>cut, median, img)', out=out)

	# not_spot = np.logical_not(spot)

	# new_img  = np.zeros(img.shape)
	# new_img[not_spot]  = img[not_spot]
	# new_img[spot] = median[spot]

	return out


def remove_outliers_stack(arr, size, threshold, axis=0, outliers='bright', k=1.0, out=None):

	arr = arr.swapaxes(0, axis)

	if(out is None):
		out = arr

	print('> Removing ' + outliers + ' outliers...')
	for i in tqdm(range(0, arr.shape[0]), unit=' images'):
		remove_outliers(arr[i], size, threshold, outliers, k, out=out[i])

	return out
