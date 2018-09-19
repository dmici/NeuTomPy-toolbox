import numpy as np
import numexpr as ne
import logging

__author__  = "Davide Micieli"
__all__     = ['convert_to_uint']


logs = logging.getLogger(__name__)

def convert_to_uint(arr, nbit=16, down=None, up=None, crop_roi=None):
	"""
	This function converts a floating point array to 8 or 16 bit unsigned integer array.
	The resampling can be performed in a user-specified dynamic range.
	The array can be also cropped by specifying the ROI coordinates.

	Parameters
	----------
	arr : ndarray
		The floating point array to convert to unsigned integer array.

	nbit : int
		The number of bits of the returned array. In must be `8` or `16`, for
		8-bit or 16-bit unsigned integer, respectively. Default value is `16`.

	down: float, optional
		The lower range limit. If `None`, the lower limit is computed as the
		minimum value of the input array.

	up: float, optional
		The upper range limit. If `None`, the upper limit is computed as the
		maximum value of the input array.

	crop_roi: tuple, optional
		Tuple defining the ROI of the array to crop. E.g. for a 3D stack
		(smin, smax, rmin, rmax, cmin, cmax), while for a 2D image
		(rmin, rmax, cmin, cmax). Default value is `None`, which disables
		the crop of the ROI.

	Returns
	-------
	m : ndarray
		The array resampled to 8-bit or 16-bit unsigned integer.
	"""

	if(arr.ndim != 3 and  arr.ndim != 2):
		raise ValueError('The input array must have 2 or 3 dimensions.')

	# check number of bits
	if(nbit == 8):
		tp  = np.uint8
	elif(nbit == 16):
		tp  = np.uint16
	else:
		raise ValueError('The output array type must be 8 bit or 16 bit unsigned integer.')


	if(crop_roi):

		old_shape = arr.shape

		if(arr.ndim == 3):
			smin, smax, rmin, rmax, cmin, cmax = crop_roi
			arr = arr[smin:smax, rmin:rmax, cmin:cmax].astype(np.float32)

		if(arr.ndim == 2):
			rmin, rmax, cmin, cmax = crop_roi
			arr = arr[rmin:rmax, cmin:cmax].astype(np.float32)

		logs.debug('Array cropped: original shape = %s, cropped array shape = %s', old_shape, arr.shape)

	arr = arr.astype(np.float32)

	if(down == None):
		down = arr.min()

	if(up == None):
		up = arr.max()

	down = down.astype(np.float32)
	up   = up.astype(np.float32)

	factor = np.float32(2.0**nbit - 1.0)

	arr = np.clip(arr, a_min=down, a_max=up, out=arr)
	arr = ne.evaluate('factor*(arr - down)/(up - down)', out=arr, truediv=True)

	logs.info('Array converted to %s using the dynamic range %s - %s', tp, down, up)

	return (np.round(arr)).astype(dtype=tp)
