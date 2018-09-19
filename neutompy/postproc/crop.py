import numpy as np
import numexpr as ne


__author__  = "Davide Micieli"
__all__     = ['get_circular_mask', 'circular_crop']

def get_circular_mask(nrow, ncol, radius=None, center=None):
	"""
	This function returns a boolean 2D array representing a circular mask.
	The values outside the circle are `False`, while the inner values are `True`.

	Parameters
	----------
	nrow: int
		The number of rows of the mask array.

	ncol: int
		The number of columns of the mask array.

	radius: float, optional
		The radius of the circle in pixel.
		The default value is 0.5 * min(number of rows,number of columns)

	center: tuple of float, optional (yc, xc)
		A tuple representing the coordinates of the center of the circle, i.e.: (yc, xc).
		The default value is the center of the input image.

	Returns
	-------
	mask : 2d array
		A boolean array that represents the circular mask. The values outside the circle
		are `False`, while the inner values are `True`.
	"""


	if(radius is None):
		radius = min(ncol, nrow)/2

	if(center is None):
		yc = ncol/2.0
		xc = nrow/2.0
	else:
		yc, xc = center

	ny = np.arange(ncol)
	nx = np.arange(nrow)

	x, y = np.meshgrid(nx, ny)

	mask = ( (y-yc + 0.5)**2 + (x-xc + 0.5)**2 ) < (radius)**2

	return mask



def circular_crop(img, axis=0, radius=None, center=None, cval=0):
	"""
	This function performs a circular crop of an image. The values outside the
	circle are replaced with the value cval (default cval=0).

	Parameters
	----------

	img: 2d array
		The array representing the image to crop.

	radius: float, optional
		The radius of the circle. The default value is 0.5 * min(number of rows,number of columns)

	center: tuple of float, optional (yc, xc)
		A tuple representing the coordinates of the center of the circle, i.e.: (yc, xc).
		The default value is the center of the input image.

	cval: float, optional
		The value used to fill the region outside the circle.

	Returns
	-------

	out: 2d array
		The cropped image.
	"""

	if (img.ndim == 3):
		img = img.swapaxes(0,axis)
		nslice, nrow, ncol = img.shape
	elif (img.ndim == 2):
		nrow, ncol = img.shape
	else:
		raise ValueError("The input array must have 2 or 3 dimensions.")

	# set type of cval
	cval =	np.array(cval, dtype=img.dtype)

	# get the circular mask	-> True inside the circle
	mask = get_circular_mask(nrow, ncol, radius, center)

	# apply the mask
	ne.evaluate('where(mask, img, cval)' , out=img)

	if (img.ndim == 3):
		img = img.swapaxes(0,axis)

	return img
