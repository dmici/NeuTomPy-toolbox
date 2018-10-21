import numpy as np
import numexpr as ne
import matplotlib.pyplot as plt
from skimage.measure import profile_line, compare_ssim
from numpy.linalg import norm
from read_roi import read_roi_file
from neutompy.postproc.crop import *
from neutompy.image.image import get_rect_coordinates_from_roi
import logging
import os
from scipy.optimize import curve_fit
from scipy.special import erf

logs = logging.getLogger(__name__)

__author__  = "Davide Micieli"
__all__     = ['CNR', 'NRMSE', 'SSIM', 'FWHM', 'get_line_profile']


def CNR(img, croi_signal=[], croi_background=[], froi_signal=[], froi_background=[]):
	"""
	This function computes the Contrast-to-Noise Ratio (CNR) as reported in
	the equation (2.7) of [1]_.
	The ROI of the signal and the background can be defined using two lists of
	coordinates or two ImageJ .roi files.

	Parameters
	----------
	img :  2d array
		The array representing the image.

	croi_signal : list
		List that contains the following coordinate of the signal roi: [rowmin, rowmax, colmin, colmax].

	croi_background : list
		List that contains the following coordinate of the background roi: [rowmin, rowmax, colmin, colmax].

	froi_signal : string
		Path of the imagej file containing the rectangular ROI of the signal.

	froi_background : string
		Path of the imagej file containing the rectangular ROI of the background.

	Returns
	-------
	CNR : float
		The CNR value computed using the ROIs given.

	References
	----------
	.. [1] D. Micieli et al., A comparative analysis of reconstruction methods
	 applied to Neutron Tomography, Journal of Instrumentation, Volume 13,
	 June 2018.
	"""

	if(img.ndim != 2):
		raise ValueError("The input array must have 2 dimensions.")

	if(croi_signal and froi_signal):
		raise ValueError("Only one method to define the ROI is accepted. Please pass croi_singal or froi_signal.")

	if(croi_background and froi_background):
		raise ValueError("Only one method to define the ROI is accepted. Please pass croi_background or froi_background.")

	if(croi_signal):
		rowmin, rowmax, colmin, colmax = croi_signal
	if(froi_signal):
		rowmin, rowmax, colmin, colmax = get_rect_coordinates_from_roi(froi_signal)

	signal = img[rowmin:(rowmax+1),  colmin:(colmax+1)]

	if(croi_background):
		rowmin, rowmax, colmin, colmax = croi_background
	elif(froi_background):
		rowmin, rowmax, colmin, colmax = get_rect_coordinates_from_roi(froi_background)

	background = img[rowmin:(rowmax+1),  colmin:(colmax+1)]

	cnr_val = ( signal.mean() - background.mean() ) / background.std()

	return cnr_val




def get_line_profile(image, start=(), end=(), froi='', ShowPlot=True, PlotTitle='Profile', linewidth=1, order=1, mode='constant', cval=0.0):
	"""
	This function returns the intensity profile of an image measured along a line defined by the points:

			start = (x_start, y_start)  [i.e. (col_start row_start)]
			end   = (x_end, y_en)       [i.e. (col_end row_end)]

	or an ImageJ .roi file containing the line selection. A plot representing the intensity profile can be shown.

    Parameters
    ----------
    image : ndarray
        The image grayscale (2D array) or a stack of images (3d array) with shape (slices, rows, columns).
        Ffor a 3D array the first axis represents the image index.

    start : 2-tuple of numeric scalar (float or int) (x y) [i.e. (col row)]
        The start point of the scan line.

    end : 2-tuple of numeric scalar (float or int) (x y) [i.e. (col row)]
        The end point of the scan line. The destination point is *included*
        in the profile, in constrast to standard numpy indexing.

    froi : string
        Path of the imagej file containing the line selection.

    ShowPlot : bool
        If True a canvas is created representing the Plot Profile.

    linewidth : int, optional
        Width of the scan, perpendicular to the line

    order : int in {0, 1, 2, 3, 4, 5}, optional
        The order of the spline interpolation to compute image values at
        non-integer coordinates. 0 means nearest-neighbor interpolation.

    mode : {'constant', 'nearest', 'reflect', 'mirror', 'wrap'}, optional
        How to compute any values falling outside of the image.

    cval : float, optional
        If `mode` is 'constant', what constant value to use outside the image.

    Returns
    -------
    return_value : array
        The intensity profile along the scan line. The length of the profile
        is the ceil of the computed length of the scan line.
	"""

	if(froi and (start or end)):
		raise ValueError('Please indicate an ImageJ .roi file or two tuples of coordinates.')

	if(froi):
		# read roi file
		roi = read_roi_file(froi)

		# convert keys to list
		l = list(roi.keys())

		# extract type
		tp = roi[l[0]]['type']

		if(tp=='line'):
			start = np.floor([roi[l[0]]['x1'], roi[l[0]]['y1']])
			end   = np.floor([roi[l[0]]['x2'], roi[l[0]]['y2']])
		else:
			raise ValueError("File specified does not contain a line selection.")


	start = start[::-1]
	end   = end[::-1]

	if(image.ndim == 3):
		image = image.swapaxes(0,2).swapaxes(0,1)

	y = profile_line(image, start, end, linewidth, order, mode, cval)


	if(ShowPlot):
		plt.figure(PlotTitle);
		plt.plot(y, '-');
		plt.show(block=False)

	if(image.ndim == 3):
		y = y.swapaxes(0,1)

	return y



def NRMSE(img, ref,  mask='whole'):
	"""
	This function computes the Normalized Root Mean Square Error (see eq. 2.9
	in [1]_) of an image respect to a reference image.

	Parameters
	----------
	img: 2d array
		Test image

	ref: 2d array
		Reference image

	mask:
		It represents the type of the ROI where the NRMSE is computed.
		You can specify: 'whole'  -> full image
						 'circ'   -> circular mask
						 'custom' -> custom mask, defined as a boolean matrix of the same shape of the images (img and ref)
	mask: 2d bool array
		Curstom mask of the same shape of the images, where the NRMSE is computed.

	Returns
	-------
	NRMSE: float
		The NRMSE value computed within the roi specified.
	"""

	if(img.shape != ref.shape):
		raise ValueError('The two input arrays must have the same shape.')

	if(img.ndim != 2 or ref.ndim != 2 ):
		raise ValueError('The two input arrays must be 2D.')

	if (type(mask) is str):

		if(mask.endswith('.roi')):

			if not os.path.isfile(mask):
				raise ValueError("The specified file .roi does not exixst.")

			rowmin, rowmax, colmin, colmax = get_rect_coordinates_from_roi(mask)
			vimg = img[rowmin:(rowmax+1),  colmin:(colmax+1)]
			vref = ref[rowmin:(rowmax+1),  colmin:(colmax+1)]

		elif((mask == 'whole') or (mask is None)):

			vimg = img
			vref = ref

		elif(mask == 'circ'):
			nrow, ncol = img.shape
			mask = get_circular_mask(nrow, ncol)
			vimg = img[mask]
			vref = ref[mask]
		else:
			raise ValueError('The mask variable must be a string representing a file path or the keywords `whole`, `circ`.')

	elif(type(mask) is np.ndarray):

		if(mask.dtype is not np.dtype('bool')):
			raise ValueError('The mask must be a boolean matrix.')

		vimg = img[mask]
		vref = ref[mask]

	else:
		raise ValueError('The type of the region-of-interest for the NRMSE computation is not valid')

	vimg = vimg.astype(np.float64)
	vref = vref.astype(np.float64)

	val = norm(vimg - vref) / norm(vref)

	return val


def sigmoid_gaus(x, k0, k1, k2, k3):
	"""
	This function computes a general formula of the Gauss error function.
	The exact formula implemented is f(x) = 0.5 * k0 * (Erf(k1*(x - k2)) + 1.0) + k3,
	where Erf is the  Gauss error function, k0, k1, k2, k3 and k4 are constants.

	Parameters
	----------
	x : 1d array
		The vector containing the abscissa values.

	k0 : float
		The height of the sigmoid function.

	k1 : float
		This parameters set the steepness of the sigmoid.

	k2 : float
		The midpoint of the sigmoid

	k3 : float
		The bias parameter.

	Returns
	-------
	val : float
		The function value.
	"""
	val = 0.5 * k0 * (erf(k1*(x - k2)) + 1.0) + k3
	return val



def FWHM(profile, yerr=None):
	"""
	This functions computes an edge quality metric from a profile of a sharp edge.
	The profile is fitted with a generic Gauss sigmoid function.
	The fitting function was then differentiated and the FWHM of the gaussian
	obtained is returned by the function.
	This method is described in detail in [1]_.

	Parameters
	----------
	profile : 1d array
		The line profile of the sharp edge.

	yerr : 1d array, optional
		The vector containing the standard deviation of the edge profile.
		If not specified the standard deviation of each point is assumed equal
		to 1.0.

	Returns
	-------
	fwhm : float
		The FWHM mean value.

	fwhm_err : float
		The FWHM error value

	profile_fitted: 1d array
		The Gauss sigmoid function evaluated for the fitting parameters.

	popt : list
		List containing the fitting parameters.

	perr : list
		List containing the errors of the fitting parameters.

	"""

	npoints = profile.size
	xdata   = np.arange(npoints)
	pin    = [None]*4
	pin[0] = profile.max() - profile.min()
	pin[1] = 0.5
	pin[2] = npoints/2
	pin[3] = profile.min()


	popt, pcov = curve_fit(sigmoid_gaus, xdata, profile, p0=pin, sigma=yerr)
	perr = np.sqrt(np.diag(pcov))

	p1     = popt[1]
	p1_err = perr[1]

	k = 2.0 * np.sqrt( np.log(2) )

	fwhm     = k / p1
	fwhm_err = k*p1_err / (p1**2)

	profile_fitted = sigmoid_gaus(xdata, *popt)

	return fwhm, fwhm_err, profile_fitted, popt, perr


def SSIM(img1, img2, circ_crop=True, L=None, K1=0.01, K2 = 0.03, sigma=1.5, local_ssim=False):
	"""
	This function computes the Structural Similarity Index (SSIM) [2]_.
	It returns global SSIM value and, optionally, the local SSIM map.

	Parameters
	----------
	img1: 2d array
		The first image to compare. SSIM index satisfies the condition of simmetry: SSIM(img1, img2) = SSIM(img2, img1)

	img2: 2d array
		The second image to compare. SSIM index satisfies the condition of simmetry: SSIM(img1, img2) = SSIM(img2, img1)

	circular_crop : bool, optional
		If True (default) the images are cropped with a circular mask,
		otherwise the SSIM is computed over the entire image.

	L : float, optional
		The data range of the input images (distance between minimum and
		maximum possible values). By default, this is estimated from
		the image data-type.

	K1 : float, optional
		A constant that prevents the division by zero (see [2]_).

	K2 : float, optional
		A constant that prevents the division by zero (see [2]_).

	sigma : float, optional
		The standard deviation of the Gaussian filter. This parameter
		sets the minimum scale at which the quality is evaluated.

	local_ssim: float, optional
		If True, the function returns the local SSIM map.

	Returns
	-------
	ssim : float
		The global SSIM index.

	map : 2d array
		The bidimendional map of the local SSIM index. This is only returned
		if `local_ssim` is set to True.

	References
	----------
	.. [2] Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P.
		(2004). Image quality assessment: From error visibility to
		structural similarity. IEEE Transactions on Image Processing,
		13, 600-612.
		https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf
	"""

	if(img1.shape != img2.shape):
		raise ValueError('The input images must have the same shape.')

	vimg1 = np.zeros(img1.shape)
	vimg2 = np.zeros(img2.shape)

	if(circ_crop):
		nrow, ncol = img1.shape
		mask = get_circular_mask(nrow, ncol)
		vimg1[mask] = img1[mask]
		vimg2[mask] = img2[mask]
	else:
		vimg1 = img1
		vimg2 = img2

	val = compare_ssim(vimg1, vimg2, data_range=L, gaussian_weights=True, sigma=sigma, k1=K1, K2=K2, use_sample_covariance=False, full=local_ssim)
	return val
