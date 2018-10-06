import numpy as np
from numpy import sin, cos
from mkl_fft import fft, ifft
from numpy.fft import fftshift, ifftshift
import pywt
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
from time import sleep

__author__  = "Davide Micieli"
__all__     = ['draw_ROI',
			   'normalize_proj',
			   'log_transform',
			   'find_COR',
			   'correction_COR',
			   'remove_outliers',
			   'remove_outliers_stack',
			   'remove_stripe',
			   'remove_stripe_stack'
			  ]

def draw_ROI(img, title, ratio=0.85):
	"""
	This function allows to select interactively a rectangular region of interest
	(ROI) over an image. The function returns the ROI coordinates.

	Parameters
	----------
	img : 2d array
		The image on which the dose roi is drawn.

	title : str
		String defining the title of the window shown.

	ratio : float, optional
		The filling ratio of the window respect to the screen resolution.
		It must be a number between 0 and 1. The default value is 0.85.

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

	if (img.ndim != 2):
		raise ValueError("The image array must be two-dimensional.")

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
	img = img/(mu+2*s)  # normalization can be improved
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
	"""
	This function computes the normalization  of the projection data using dark
	and flat images by performing the ratio: (proj - dark) / (flat - dark).
	The ratio is computed using the mean or the median of the dark and flat
	images (defined by the mode). The division by zero is prevented by assigning
	to the denominator a small value (min_denom) if (flat - dark  == 0). The
	result can be stored in an output array by using the 'out' parameter. If
	same as proj, the computation is done in place.
	"""

	if(min_denom<=0.0):
		raise ValueError('The parameter min_ratio must be positive.')

	if (mode == 'mean'):
		func = np.mean
	elif(mode == 'median'):
		func = np.median
	else:
		raise ValueError('Not valid function for projecting flat and dark images\
		 				  along z axis. Set mean or median')

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
	"""
	This function computes the normalization of the projection data using dark
	and flat images. If the source intensity is not stable the images can
	be normalized respect to the radiation dose (see formula 2.2 in [1]_).
	In this case, the user must specify a region (the dose ROI) where the sample
	never appears. If not interested in reconstructing the entire field-of-view,
	the normalization can be performed using only a region of interest (crop ROI)
	of all projections.
	The dose ROI and the crop ROI can be drawn interactively on the image or
	specified using the index coordinates or an ImageJ .roi file. A stack of dark
	and flat images is required and the median or the mean of the stack is
	used to compute the main fomula.

	Parameters
	----------
	proj : ndarray
		A three-dimensional stack of raw projections.
		The 0-axis represents theta.

	dark: ndarray
		A three-dimensional stack of dark-field images.

	flat : ndarray
		A three-dimensional stack of flat-field images.

	proj_180: 2d arrays, optional
		The projection at 180 degree. Specify it only if it is not already
		included in the stack `proj`. It is disabled by default (None).

	out : ndarray, optional
		The output array returned by the function. If it is the same as proj,
		the computation will be done in place.

	dose_file : str, optional
		String defining the path or the name of the Image ROI file that includes
		the dose ROI.

	dose_coor : tuple, optional
		Tuple defining the indexes range for each axis of the ROI dose.
		Specify it in this way: ``(row_start, row_end, col_start, col_end)``

	dose_draw : bool, optional
		If True the dose ROI is selected interactively on the image by the user.
		The default is True.

	crop_file : str, optional
		String defining the path or the name of the Image ROI file that includes
		the ROI to crop.

	crop_coor : tuple, optional
		Tuple defining the indexes range, for each axis, of the ROI to crop.
		Specify it in this way: ``(row_start, row_end, col_start, col_end)``

	crop_draw : bool, optional
		If True the ROI to crop is selected interactively on the image by the user.
		The default is True.

	min_denom : float, optional
		Minimum permitted value of the denominator. It must be a small number
		that prevents the division by zero. Defalut value is ``1.0e-6``.

	min_ratio : float, optional
		Minimum permitted value of normalized projections. It must be a small
		positive number that prevents negative values normalized data.
		Defalut value is ``1.0e-6``.

	max_ratio : float, optional
		Maximum permitted value of normalized projections. It must be a
		positive number. It mitigates the magnitude of the bright outliers within
		normalized data. Defalut value is ``10``.

	log : bool, optional
		If ``True`` the log-transform of the normalized data is performed. If
		``False``, the normalized data without log-transform are returned.
		Default value is ``False``.

	mode : string, optional
	 	If `dose_draw` or `crop_draw` is ``True`` the user can select interactively
		the ROI. A window showing a representative image of the projection stack
		is created. This image can be the mean or the standard deviation computed
		pixel-wise over the projection stack. Hence, allowed values of `mode` are
		``mean`` and ``std``. Default value is ``mean``.

	sino_order : bool, optional
		If ``True`` a stack of sinograms is returned (0 axis represents the
		projections y-axis). If ``False`` a stack of projections is returned
		(0 axis represents theta). Default value is ``False``.

	Returns
	-------
	out : ndarray
		Three-dimensional stack of the normalized projections.

	out_180 : 2d arrays
		The normalized projection at 180 degree. It is returned only
		if `proj_180` is an array.

	References
	----------
	.. [1] D. Micieli et al., A comparative analysis of reconstruction methods
	 applied to Neutron Tomography, Journal of Instrumentation, Volume 13,
	 June 2018.

	Examples
	--------
	Normalize dataset selecting interactively the ROI to crop and the dose ROI.

	>>> import neutompy as ntp
	>>> norm =  ntp.normalize_proj(proj, dark, flat, dose_draw=True, crop_draw=True)

	Normalize dataset and the raw projection at 180 degree:

	>>> fname  = ntp.get_image_gui('', message = 'Select raw projection at 180°...')
	>>> img180 = ntp.read_image(fname)
	>>> norm, norm_180 = ntp.normalize_proj(proj, dark, flat, proj_180=img180)

	Normalize dataset using two ImageJ .roi file to define the ROI to crop and
	the dose ROI:

	>>> norm = ntp.normalize_proj(proj, dark, flat, dose_file='./dose.roi', crop_file='./crop.roi'
							 dose_draw=False, crop_draw=False)

	Normalize the dataset with the log-transform:

	>>> norm = ntp.normalize_proj(proj, dark, flat, log=True)

	Trivial data normalization using the whole field of view and without the
	dose correction:

	>>> norm = ntp.normalize_proj(proj, dark, flat, dose_draw=False, crop_draw=False)
	"""
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
		raise ValueError('Not valid value of the variable show_opt. Choose "mean"\
		 or "std" to show the mean or the standard deviation computed pixel-wise over\
		 all projections.')

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
			print("> Crop a ROI of the projections to reconstruct...")
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
			print("> Background ROI selection. Select a region free of the sample...")
			rmin_d, rmax_d, cmin_d, cmax_d  = draw_ROI(show_proj, 'Select a region free of the sample...' )

		ds_roi = rmin_d, rmax_d, cmin_d, cmax_d


		if(mode=='mean'):
			func = np.mean
		elif(mode=='median'):
			func = np.median
		else:
			raise ValueError('Not valid function for projecting flat and dark images along z axis.\
							Set mean or media.')

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

	if (proj_c.shape[0] != 1):
		print('> Normalization...')
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
	This function computes the minus log of an array. In trasmission CT the input\
	array must be the normalized dataset after flat-fielding correction.

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
	imgpad_size = np.array([ int(round(abs(l1*cos(th)) + abs(l2*sin(th)))),
	 						int(round(abs(l2*cos(th)) + abs(l1*sin(th)))) ])


	before =  (imgpad_size - np.array(img.shape))//2
	after  =  imgpad_size -  np.array(img.shape) -before

	before = tuple(before)
	after  = tuple(after)

	imgpad = np.pad(img, pad_width=((before[0], after[0]),  (before[1], after[1])), mode='edge')


	out = rotate_2(imgpad, theta, interpolator)

	return out[ before[0]:(l1+before[0]), before[1]:(l2 + before[1])]



def find_COR(proj_0, proj_180, nroi=None, ref_proj=None, ystep=5, ShowResults=True):
	"""
	This function estimates the offset and the tilt angle of the rotation axis
	respect to the detector using the projections at 0 and at 180 degree. The user
	selects interactively different regions where the sample is visible. Once the
	position of the rotation axis is found the results are shown in two figures.
	In the first are shown the computed rotation axis and the image obtained as
	proj_0 - pro_180[:,::-1] (the projection at 180 is flipped horizontally)
	before the correction.
	The second figure shows the difference image proj_0 - pro_180[:,::-1] after
	the correction and the histogram of abs(proj_0 - pro_180[:,::-1]).

	Parameters
	----------
	proj_0 : 2d array
		The projection at 0 degrees.

	proj_180 : 2d array
		The projection at 180 degrees.

	nroi : int, optional
		The number of the region of interest to select  for the computation
		of the rotation axis position. Default is None, hence the value is read
		as input from keyboard.

	ref_proj: 2d array
		The image shown to select the region of interest. Default is None, hence
		proj_0 is shown.

	ystep: int, optional
		The center of rotation position is computed every `ystep`. Default value
		is 5.

	ShowResults: bool, optional
		If True, the the two figures that summarize the result are shown.
		Default is True.

	Returns
	-------
	middle_shift : float
		The horizontal shift of the rotation axis respect to the center of the
		detector.

	theta : float
		The tilt angle formed by the rotation axis and the vertical axis of the
		detector.
	"""
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
		print('To compute the rotation axis position it is necessary to select one or multiple regions where the sample is present.\nHence you must draw the different regions vertically starting from top to bottom.')
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


	print("Rotation axis Found!")
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

	plt.title('$P_0 - P^{flipped}_{\pi}$ before correction')
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
	plt.title('$P_0 - P^{flipped}_{\pi}$ after correction')
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


def correction_COR(norm_proj, proj_0, proj_180, show_opt='mean', shift=None,
				theta=None, nroi=None, ystep=5):
	"""
	This function corrects the misalignment of the rotation axis respect to the
	vertical axis of the detector. The user can choose to insert manually the offset
	and the tilt angle of the rotation axis	respect to the detector, if known
	parameters, or to estimate them using the projections at 0 and at 180 degrees.
	In this case, the user selects interactively different regions where the
	sample is visible. Once the position of the rotation axis is found the
	results are shown in two figures. In the first are shown the estimated rotation
	axis and the image obtained as proj_0 - pro_180[:,::-1] (projection at 180
	is flipped horizontally) before the correction.
	The second figure shows the difference image (proj_0 - pro_180[:,::-1]) after
	the correction and the histogram of abs(proj_0 - pro_180[:,::-1]).

	Parameters
	----------
	norm_proj : ndarray
		A stack of projections. The 0-axis represents theta.

	proj_0 : 2d array
		The projection at 0 degrees.

	proj_180 : 2d array
		The projection at 180 degrees.

	show_opt : str, optional
		A string defining the image to show for the selection of the ROIs.

		Allowed values are:

		``mean`` -> shows the mean of `norm_proj` along the O-axis.

		``std``  -> shows the standard deviation of `norm_proj` along the O-axis

		``zero`` -> shows proj_0

		``pi``   -> shows proj_180

		Default value is ``mean``.

	shift: int, optinal
		The horizontal shift in pixel of the rotation axis respect to the
		vertical axis of the detector. It can be specified if known parameter.
		The default is None, hence this parameter is estimated from the projections.

	theta: int, optinal
		The tilt angle in degrees of the rotation axis respect to the vertical
		axis of the detector. It can be specified if known parameter.
		The default is None, hence this parameter is estimated from the projections.

	nroi : int, optional
		The number of the region of interest to select  for the computation
		of the rotation axis position. Default is None, hence the value is read
		as input from keyboard.

	ystep: int, optional
		The center of rotation position is computed every `ystep`. Default value
		is 5.

	Returns
	-------
	norm_proj : ndarray
		The stack after the correction. The computation is done in place.
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
			sleep(0.5)

			while True:
					ans = input('> Rotation axis found. Do you want to correct all projections?\
					 \n[Y] Yes, correct them.  \n[N] No, find it again.\
					 \n[C] Cancel and abort the script.\
					 \nType your answer and press [Enter] :')
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
	print('> Correcting rotation axis misalignment...')
	for s in tqdm(range(0, norm_proj.shape[0]), unit=' images'):
		# norm_proj[s,:,:] = np.roll(rotate(norm_proj[s,:,:], theta, preserve_range=True, order=1, mode='edge'), shift, axis=1)
		norm_proj[s,:,:] = np.roll(rotate_sitk(norm_proj[s,:,:], theta, interpolator=sitk.sitkLinear), shift, axis=1)

	return norm_proj


def median_filter(img, radius):
	"""
	This function applies a median filter on an image.

	Parameters
	----------
	img : 2d array
		The image to filter. It must be a two-dimensional array.

	radius: int or tuple of int
		The radius of the neighborhood. The radius is defined separately for
		each dimension as the number of pixels that the neighborhood extends
		outward from the center pixel.

	Returns
	-------
		The filtered image.
	"""
	if (img.ndim != 2):
		raise ValueError('The input iamge must be two-dimensional.')

	simg = sitk.GetImageFromArray(img)
	mf = sitk.MedianImageFilter()
	mf.SetRadius(radius)
	fimg = sitk.GetArrayFromImage(mf.Execute(simg))

	return fimg

def median_filter_stack(arr, radius, axis=0):
	"""
	This function applies a 2D median filter on a stack of images.

	Parameters
	----------
	img : ndarray
		The stack to filter. It must be a three-dimensional array.

	radius: int or tuple of int
		The radius of the neighborhood. The radius is defined separately for
		each dimension as the number of pixels that the neighborhood extends
		outward from the center pixel.

	axis : int
		The axis along which the 2D median filter is iterated.

	Returns
	-------
	out : ndarray
		The filtered stack of images.
	"""
	if (arr.ndim != 3):
		raise ValueError('The input array must be three-dimensional.')

	arr = arr.swapaxes(0, axis)

	for i in tqdm(range(0, arr.shape[0]), unit=' images'):
		arr[i] = median_filter(arr[i], radius)

	arr = arr.swapaxes(0, axis)

	return arr


def convolution_2D (img, kernel, boundary='ZERO_PAD', out_mode='SAME'):
	"""
	Perform a 2D convolution with an arbitrary kernel.

	Parameters
	----------
	img : 2d array
		The input image.

	kernel : 2d array
		The kernel to use for the convolution.

	boundary: str
		The boundary padding type. Allowed values are:
		``ZERO_PAD``, ``ZERO_FLUX_NEUMANN_PAD`` and ``PERIODIC_PAD``.

	out_mode: str
		Sets the output region mode. If set to `SAME`, the output region
		will be the same as the input region, and regions of the image
		near the boundaries will contain contributions from outside the
		input image as determined by the boundary condition set in
		`boundary`. If set to `VALID`, the output region
		consists of pixels computed only from pixels in the input image
		(no extrapolated contributions from the boundary condition are
		needed). The output is therefore smaller than the input
		region. Default value is `SAME`.

	Returns
	-------
	out : 2d array
		The filtered image. The type of the array is inferred
		from img and kernel type.
	"""

	simg    = sitk.GetImageFromArray(img)
	skernel = sitk.GetImageFromArray(kernel)

	conv = sitk.ConvolutionImageFilter()
	conv.SetBoundaryCondition(eval('conv.' + boundary))
	conv.SetOutputRegionMode(eval('conv.' + out_mode))

	filt = sitk.GetArrayFromImage(conv.Execute(simg, skernel))

	return filt

def std_map(img, radius):

	if (type(radius) is not int or radius==0):
		raise ValueError('Kernel radius must be a positive integer.')

	vartype = img.dtype # get the array type
	img     = img.astype(vartype)
	img2    = img**2
	ones    = np.ones(img.shape, dtype=vartype)
	kernel = np.ones((2*radius+1, 2*radius+1),  dtype=vartype)

	s  = convolution_2D(img, kernel)
	s2 = convolution_2D(img2, kernel)
	ns = convolution_2D(ones, kernel)

	return np.sqrt((s2 - s**2 / ns) / (ns-1))


def remove_outliers(img, radius, threshold, outliers='bright', k=1.0, out=None):
	"""
	This function removes bright and dark outliers from an image.
	It replaces a pixel by the median of the pixels in the neighborhood
	if it deviates from the median by more than a certain value (k*threshold).
	The threshold can be specified by the user as a global value or computed as
	the local standard deviation map.

	Parameters
	----------
	img : 2d array
		The image to elaborate.

	radius: int or tuple of int
		The radius of the neighborhood. The radius is defined separately for
		each dimension as the number of pixels that the neighborhood extends
		outward from the center pixel.

	threshold: float or str
		If it is the string 'local' the local standard deviation map is taken
		into account. Conversely, if it is a float number the threshold is global.

	outliers: str, optional
		A string defining the type of outliers to remove. Allowed values are
		``bright`` and ``dark``. Default is `bright`.

	k : float, optional
		A pixel is replaced by the median of the pixels in the neighborhood
		if it deviates from the median by more than k*threshold.
		Default value is 1.0.

	out: 2d array, optional
		If same as img, then the computation is done in place. Default is None,
		hence this behaviour is disabled.

	Returns
	-------
	out : 2d array
		The image obtained by removing the outliers.
	"""
	if not (threshold=='local' or type(threshold) in [float, int]):
		raise ValueError('The threshold must be a float variable or the string `local`.')

	if(threshold=='local'):
		threshold = std_map(img, radius)

	median = median_filter(img, radius)

	if(outliers=='bright'):
		diff    = img - median
		spot    = diff > threshold * k
	elif(outliers=='dark'):
		diff    =  median - img
		spot    = diff   > threshold * k

	# convert type
	cut = np.array(threshold*k, dtype = img.dtype)

	out = ne.evaluate('where(diff>cut, median, img)', out=out)

	return out


def remove_outliers_stack(arr, radius, threshold, axis=0, outliers='bright', k=1.0, out=None):
	"""
	This function removes bright and dark outliers from a stack of images.
	The algorithm elaborates 2d images and the filtering is iterated over all
	images in the stack.
	The function replaces a pixel by the median of the pixels in the 2d neighborhood
	if it deviates from the median by more than a certain value (k*threshold).
	The threshold can be specified by the user as a global value or computed as
	the local standard deviation map.

	Parameters
	----------
	arr : ndarray
		The stack to elaborate.

	radius: int or tuple of int
		The radius of the 2D neighborhood. The radius is defined separately for
		each dimension as the number of pixels that the neighborhood extends
		outward from the center pixel.

	threshold: float or str
		If it is the string 'local' the local standard deviation map is taken
		into account. Conversely, if it is a float number the threshold is global.

	axis : int
		The axis along wich the outlier removal is iterated.

	outliers: str, optional
		A string defining the type of outliers to remove. Allowed values are
		``bright`` and ``dark``. Default is `bright`.

	k : float, optional
		A pixel is replaced by the median of the pixels in the neighborhood
		if it deviates from the median by more than k*threshold.
		Default value is 1.0.

	out: 2d array, optional
		If same as arr, then the computation is done in place. Default is None,
		hence this behaviour is disabled.

	Returns
	-------
	out : ndarray
		The array obtained by removing the outliers.
	"""
	arr = arr.swapaxes(0, axis)

	if(out is None):
		out = arr

	print('> Removing ' + outliers + ' outliers...')
	for i in tqdm(range(0, arr.shape[0]), unit=' images'):
		remove_outliers(arr[i], radius, threshold, outliers, k, out=out[i])

	out = out.swapaxes(0, axis)
	return out


def remove_stripe(img, level, wname='db5', sigma=1.5):
	"""
	Suppress horizontal stripe in a sinogram using the Fourier-Wavelet based
	method by Munch et al. [2]_.

	Parameters
	----------
	img : 2d array
		The two-dimensional array representig the image or the sinogram to de-stripe.

	level : int
		The highest decomposition level.

	wname : str, optional
		The wavelet type. Default value is ``db5``

	sigma : float, optional
		The damping factor in the Fourier space. Default value is ``1.5``

	Returns
	-------
	out : 2d array
		The resulting filtered image.

	References
	----------
	.. [2] B. Munch, P. Trtik, F. Marone, M. Stampanoni, Stripe and ring artifact removal with
		   combined wavelet-Fourier filtering, Optics Express 17(10):8567-8591, 2009.
	"""

	nrow, ncol = img.shape

	# wavelet decomposition.
	cH = []; cV = []; cD = []

	for i in range(0, level):
		img, (cHi, cVi, cDi) = pywt.dwt2(img, wname)
		cH.append(cHi)
		cV.append(cVi)
		cD.append(cDi)

	# FFT transform of horizontal frequency bands
	for i in range(0, level):
		# FFT
		fcV=fftshift(fft(cV[i], axis=0))
		my, mx = fcV.shape

		# damping of vertical stripe information
		yy2  = (np.arange(-np.floor(my/2), -np.floor(my/2) + my))**2
		damp = - np.expm1( - yy2 / (2.0*(sigma**2)) )
		fcV  = fcV * np.tile(damp.reshape(damp.size, 1), (1,mx))

		#inverse FFT
		cV[i] = np.real( ifft( ifftshift(fcV), axis=0) )

	# wavelet reconstruction
	for i in  range(level-1, -1, -1):
		img = img[0:cH[i].shape[0], 0:cH[i].shape[1]]
		img = pywt.idwt2((img, (cH[i], cV[i], cD[i])), wname)

	return img[0:nrow, 0:ncol]


def remove_stripe_stack(arr, level, wname='db5', sigma=1.5, axis=1, out=None):
	"""
	Suppress horizontal stripe in a stack of sinograms or a stack of projections
	using the Fourier-Wavelet based method by Munch et al. [3]_ .

	Parameters
	----------
	arr : 3d array
		The tomographic data. It can be a stack of projections (theta is the 0-axis)
		or a stack of images (theta is the 1-axis).

	level : int
		The highest decomposition level

	wname : str, optional
		The wavelet type. Default value is ``db5``

	sigma : float, optional
		The damping factor in the Fourier space. Default value is ``1.5``

	axis : int, optional
		The axis index of the theta axis.
		Default value is ``1``.

	out : None or ndarray, optional
		The output array returned by the function. If it is the same as `arr`,
		the computation will be done in place.
		Default value is None, hence a new array is allocated and returned by
		the function.

	Returns
	-------
	outarr : 3d array
		The resulting filtered dataset.

	References
	----------
	.. [3] B. Munch, P. Trtik, F. Marone, M. Stampanoni, Stripe and ring artifact removal with
		   combined wavelet-Fourier filtering, Optics Express 17(10):8567-8591, 2009.
	"""

	if(arr.ndim != 3):
		raise ValueError('Input array must be three-dimensional')

	if(out is None):
		out_arr = np.zeros(arr.shape, dtype=arr.dtype)

	if isinstance(out, np.ndarray):
		out_arr = out

	arr     = arr.swapaxes(0, axis)
	out_arr = out_arr.swapaxes(0,axis)

	print('Removing stripe...')
	for i in tqdm(range(0, arr.shape[0]), unit='images'):
		out_arr[i] = remove_stripe(arr[i], level, wname, sigma)


	out_arr = out_arr.swapaxes(0,axis)

	return out_arr
