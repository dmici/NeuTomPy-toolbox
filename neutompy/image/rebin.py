import numpy as np
import logging

__author__  = "Davide Micieli"
__all__     = ['rebin']


logs = logging.getLogger(__name__)


def rebin3D(arr, binsize, dtype=np.float32):
	"""
	This function rebins a 3D stack of images by averaging.

	Parameters
	----------
	arr :  ndarray
		The 3D stack of images to rebin.

	binsize: tuple
		Tuple defining the bin size for each axis of the 3D array.

	dtype: type, optional
		The type of the rebbined array.

	Returns
	-------
	m : ndarray
		The rebinned stack.
	"""
	arr = arr.astype(np.float32)

	shape = ( arr.shape[0] // binsize[0], binsize[0],
			  arr.shape[1] // binsize[1], binsize[1],
			  arr.shape[2] // binsize[2], binsize[2])

	resto_0 = arr.shape[0]  % binsize[0]
	resto_1 = arr.shape[1]  % binsize[1]
	resto_2 = arr.shape[2]  % binsize[2]

	if  (resto_0 != 0):
		arr = arr[:-resto_0, :, :]

	if(resto_1 != 0):
		arr = arr[:, :-resto_1, :]

	if(resto_2 != 0):
		arr = arr[:, : , :-resto_2]

	return arr.reshape(shape).mean(-1).mean(1).mean(2).astype(dtype)


def rebin2D(arr, binsize, dtype=np.float32):
	"""
	This function rebins a 2D image by averaging.

	Parameters
	----------
	arr :  ndarray
		The 2D image to rebin.

	binsize: tuple
		Tuple defining the bin size for each axis of the image.

	dtype: type, optional
		The type of the rebbined array.

	Returns
	-------
	m : ndarray
		The rebinned image.
	"""

	arr = arr.astype(np.float32)

	shape = ( arr.shape[0] // binsize[0], binsize[0],
		      arr.shape[1] // binsize[1], binsize[1])

	resto_0 = arr.shape[0]  % binsize[0]
	resto_1 = arr.shape[1]  % binsize[1]

	if  (resto_0 != 0):
		arr = arr[:-resto_0, :]

	if(resto_1 != 0):
		arr = arr[:, :-resto_1]

	return arr.reshape(shape).mean(-1).mean(1).astype(dtype)


def rebin(arr, binsize, dtype=np.float32):
	"""
	This function rebins a single 2D image or 3D stack.
	It reduces the size of an image or stack of images by binning groups of pixels of user-specified sizes. The resulting pixels are computed as average.

	Parameters
	----------
	arr :  ndarray, 2d or 3d
		The 2D image or the 3D stack of images to rebin.

	binsize: tuple
		Tuple defining the bin size for each axis of the array.
		E.g.: for 2d array (bin_0, bin_1)
			  for 3d array (bin_0, bin_1, bin_2)

	dtype: type, optional
		The type of the returned rebbined array.

	Returns
	-------
	m : ndarray
		The rebinned image or stack of images.
	"""
	if(arr.ndim==3 and len(binsize)==3):
		return rebin3D(arr, binsize, dtype)

	elif (arr.ndim==2 and len(binsize)==2):
		return rebin2D(arr, binsize, dtype)

	else:
		raise ValueError("Array shape or binning shape not valid.")
