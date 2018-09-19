import numpy as np
import astra
import logging
import neutompy.recon.nnfbp as nnfbp
import neutompy.recon.mrfbp as mrfbp
from tqdm import tqdm
from neutompy.recon import optomo

logs = logging.getLogger(__name__)

__author__  = "Davide Micieli"
__all__     = ['recon_slice',
	       'recon_stack',
	       'reconstruct',
	       'get_astra_proj_matrix']

# register nn-fbp training and rec plugin
astra.plugin.register(nnfbp.plugin_prepare)
astra.plugin.register(nnfbp.plugin_rec)

# register mr-fbp plugin
astra.plugin.register(mrfbp.plugin)


def get_astra_proj_matrix(nd, angles, method):
	"""
	This function returns an object that imitates a projection matrix.

	Parameters
	----------
	nd : int
		The number of pixels within a row of the detector.

	angles : 1d array, float
		The array containing the view angles in radians.
		For example, for a uniformly spaced scan from 0 to 360 degree with a number
		of projections nangles:
		angles = np.linspace(0, 2*np.pi, nangles, endpoint=False)

	method : string
		A string defining the name of the reconstruction algorithm.

		Available CPU-based methods are:
			``BP``, ``FBP``, ``SIRT``, ``SART``, ``ART``, ``CGLS``, ``NN-FBP``,
			``NN-FBP-train``, ``MR-FBP``
		Available GPU-based methods are:
			``BP_CUDA``, ``FBP_CUDA``, ``SIRT_CUDA``, ``SART_CUDA``, ``CGLS_CUDA``

	Returns
	-------
	pmat : OpTomo object
		The object that imitates a projection matrix.
	"""
	# Create the ASTRA volume and projection geometries
	vol_geom = astra.create_vol_geom(nd, nd)
	proj_geom = astra.create_proj_geom('parallel', 1.0, nd, angles)

	# Create the ASTRA projector
	if (method.endswith('CUDA')):
		pid = astra.create_projector('cuda', proj_geom, vol_geom)    # GPU
	else:
		pid = astra.create_projector('linear', proj_geom, vol_geom)  # CPU

	# initialize the projection matrix
	#pmat = astra.OpTomo(pid)
	pmat = optomo.OpTomo(pid)

	return pmat




def recon_slice(sinogram, method, pmat, parameters=None, pixel_size=1.0, offset=0):
	"""
	This function reconstructs a single sinogram for a 2D parallel beam geaometry.

	Parameters
	----------
	sinogram : 2d array
		Array representing the sinogram to reconstruct. The rows are projections
		at different angles.

	method : str
		A string defining the name of the reconstruction algorithm.

		Available CPU-based methods are:
			``BP``, ``FBP``, ``SIRT``, ``SART``, ``ART``, ``CGLS``, ``NN-FBP``,
			``NN-FBP-train``, ``MR-FBP``
		Available GPU-based methods are:
			``BP_CUDA``, ``FBP_CUDA``, ``SIRT_CUDA``, ``SART_CUDA``, ``CGLS_CUDA``

	pmat : OpTomo object
		The ASTRA object that imitates a projection matrix.

	parameters: dict, optional
		Specific options of the reconstruction algorithm defined in `method`.
		The complete list of the available options can be found within the ASTRA toolbox
		documentation: https://www.astra-toolbox.com/docs/algs/index.html.

	pixel_size : float, optional
		The detector pixel size. If specified in cm the attenuation coefficient
		values are returned in cm^-1. Default value is 1.0.

	offset : int, optional
		The offset of the rotation axis with respect to the vertical axis of the detector.
		If offset is positive the rotation axis is at right-side of the detector
		vertical axis. If negative, is at left-side.
		Default value is 0.

	Returns
	-------
	rec : 2d array
		The reconstructed slice.

	Examples
	--------
	GPU-based FBP reconstruction with the Hamming filter:

	>>> import neutomopy as ntp
	>>> # read the sinogram
	>>> sinogram  = ntp.read_image('file.tiff')
	>>> na, nd  = sinogram.shape
	>>> angles = np.linspace(0, np.pi*2, na, endpoint=False)
	>>> p   = ntp.get_astra_proj_matrix(nd, angles, "FBP_CUDA")
	>>> rec = ntp.recon_slice(sinogram, "FBP_CUDA", p, parameters={"FilterType":"hamming"})

	The filters for FBP_CUDA reconstruction included in the ASTRA toolbox are:

	 ``ram-lak`` (default), ``shepp-logan``, ``cosine``, ``hamming``, ``hann``, ``none``, ``tukey``,
	 ``lanczos``, ``triangular``, ``gaussian``, ``barlett-hann``, ``blackman``, ``nuttall``,
	 ``blackman-harris``, ``blackman-nuttall``, ``flat-top``, ``kaiser``,
	 ``parzen``, ``projection``, ``sinogram``, ``rprojection``, ``rsinogram``.

	GPU-based SIRT reconstruction with 100 iterations and limiting the pixel values
	in the range [0,2]:

	>>> rec =  ntp.recon_slice(sinogram, "SIRT_CUDA", p,
	parameters={"iterations":100, "MinConstraint": 0.0, "MaxConstraint" = 2.0 })
	"""
	if(type(offset) is float):
		offset = round(offset)
		logs.warning('Warning: Offset must be an integer. Float variable is converted to int.')

	if parameters is None:
		parameters = {}

	if 'iterations' in list(parameters.keys()):
		iterations = parameters['iterations']
		opts = {key: parameters[key] for key in parameters if key != 'iterations'}
	else:
		iterations = 1
		opts = parameters

	if(method=='NN-FBP-train' or method == 'NN-FBP-prepare'):
		method = 'NN-FBP-prepare'
		excluded_keys  = ['hidden_nodes','filter_file']
		opts = {key: parameters[key] for key in parameters if key not in excluded_keys}

	if(method=='NN-FBP'):
		opts = {'filter_file': parameters['filter_file']}

	# sinogram normalization
	pixel_size  = float(pixel_size)
	sinogram = sinogram / pixel_size

	# circular shift if the COR axis is misaligned with respect to the detector axis
	if(offset):
		sinogram  = np.roll(sinogram, - offset, axis=1)


	rec = pmat.reconstruct( method, sinogram, iterations=iterations, extraOptions=opts )

	return rec


def recon_stack(proj, method, pmat, parameters=None, pixel_size=1.0, offset=0, sinogram_order=False):
	"""
	This function reconstructs a stack of sinograms or a stack of projections
	for a 2D parallel beam geaometry.

	Parameters
	----------
	proj : 3d array
		The data reconstruct. In can be a stack of sinograms or a
		stack of projections.

	method : str
		A string defining the name of the reconstruction algorithm.

		Available CPU-based methods are:
			``BP``, ``FBP``, ``SIRT``, ``SART``, ``ART``, ``CGLS``, ``NN-FBP``,
			``NN-FBP-train``, ``MR-FBP``
		Available GPU-based methods are:
			``BP_CUDA``, ``FBP_CUDA``, ``SIRT_CUDA``, ``SART_CUDA``, ``CGLS_CUDA``

	pmat : OpTomo object
		The ASTRA object that imitates a projection matrix.

	parameters: dict, optional
		Specific options of the reconstruction algorithm defined in `method`.
		The complete list of the available options can be found within the ASTRA toolbox
		documentation: https://www.astra-toolbox.com/docs/algs/index.html.

	pixel_size : float, optional
		The detector pixel size. If specified in cm the attenuation coefficient
		values are returned in cm^-1. Default value is 1.0.

	offset : int, optional
		The offset of the rotation axis with respect to the vertical axis of the detector.
		If offset is positive the rotation axis is at right-side of the detector
		vertical axis. If negative, is at left-side.
		Default value is 0.

	sinogram_order : bool, optional
		If ``True`` the input array is read as a stack of sinograms (0 axis
		represents the projections y-axis). If ``False`` the input array is read
		as a stack of projections (0 axis represents theta).
		Default value is ``False``.

	Returns
	-------
	rec : 3d array
		The reconstructed volume.

	Examples
	--------
	GPU-based FBP reconstruction with the Hamming filter:

	>>> import neutomopy as ntp
	>>> # read stack of sinograms
	>>> data  = ntp.read_tiff_stack('./sinograms/img_0000.tiff')
	>>> _, na, nd  = data.shape
	>>> angles = np.linspace(0, np.pi*2, na, endpoint=False)
	>>> p   = ntp.get_astra_proj_matrix(nd, angles, "FBP_CUDA")
	>>> rec = ntp.recon_stack(data, "FBP_CUDA", p, sinogram_order=True,
							  parameters={"FilterType":"hamming"})

	The filters for FBP_CUDA reconstruction included in the ASTRA toolbox are:

	 ``ram-lak`` (default), ``shepp-logan``, ``cosine``, ``hamming``, ``hann``, ``none``, ``tukey``,
	 ``lanczos``, ``triangular``, ``gaussian``, ``barlett-hann``, ``blackman``, ``nuttall``,
	 ``blackman-harris``, ``blackman-nuttall``, ``flat-top``, ``kaiser``,
	 ``parzen``, ``projection``, ``sinogram``, ``rprojection``, ``rsinogram``.

	GPU-based SIRT reconstruction with 100 iterations and limiting the values of
	the reconstructed pixel in the range [0,2]:

	>>> rec =  ntp.recon_stack(data, "SIRT_CUDA", p, sinogram_order=True,
	parameters={"iterations":100, "MinConstraint": 0.0, "MaxConstraint" = 2.0 })
	"""
	if parameters is None:
		parameters = {}

	if not sinogram_order:
		proj = np.swapaxes(proj, 0, 1)

	nslice, na, nd = proj.shape

	rec = np.zeros((nslice, nd, nd), dtype=np.float32)

	for s in tqdm(range(0, nslice), unit=' slices'):

		if(method=='NN-FBP-train' or  method=='NN-FBP-prepare'):
			parameters['z_id'] = s

		rec[s] = recon_slice(proj[s], method, pmat, parameters=parameters, pixel_size=pixel_size, offset=offset)

	if(method=='NN-FBP-train' or  method=='NN-FBP-prepare'):
		nnfbp.plugin_train(parameters['traindir'], parameters['hidden_nodes'], parameters['filter_file'])

	return rec


def reconstruct(tomo, angles, method, parameters=None, pixel_size=1.0, offset=0, sinogram_order=False):
	"""
	This function reconstructs a dataset of normalized projections or a sinogram
	for a 2D parallel beam geaometry.

	Parameters
	-----------
	tomo : 2d or 3d array
		It can be a single sinogram, a three-dimensional stack of projections
		or a three-dimensional stack of sinograms.

	angles : 1d array, float
		The array containing the view angles in radians.
		For example, for a uniformly spaced scan from 0 to 360 degree with a number
		of projections nangles:
		angles = np.linspace(0, 2*np.pi, nangles, endpoint=False)

	method : str
		A string defining the name of the reconstruction algorithm.

		Available CPU-based methods are:
			``BP``, ``FBP``, ``SIRT``, ``SART``, ``ART``, ``CGLS``, ``NN-FBP``,
			``NN-FBP-train``, ``MR-FBP``

		Available GPU-based methods are:
			``BP_CUDA``, ``FBP_CUDA``, ``SIRT_CUDA``, ``SART_CUDA``, ``CGLS_CUDA``

	parameters: dict, optional
		Specific options of the reconstruction algorithm defined in `method`.
		The complete list of the available options can be found within the ASTRA toolbox
		documentation: https://www.astra-toolbox.com/docs/algs/index.html.

	pixel_size : float, optional
		The detector pixel size. If specified in cm the attenuation coefficient
		values are returned in cm^-1. Default value is 1.0.

	offset : int, optional
		The offset of the rotation axis with respect to the vertical axis of the detector.
		If offset is positive the rotation axis is at right-side of the detector
		vertical axis. If negative, is at left-side.
		Default value is 0.

	sinogram_order : bool, optional
		If ``True`` the input array is read as a stack of sinograms (0 axis
		represents the projections y-axis). If ``False`` the input array is read
		as a stack of projections (0 axis represents theta).
		Default value is ``False``.

	Returns
	-------
	rec : 2d, 3d array
		The reconstructed volume if `tomo` is three-dimensional. The reconstructed
		slice if tomo is a single sinogram.

	Examples
	--------
	GPU-based FBP reconstruction with the Hamming filter:

	>>> import neutomopy as ntp
	>>> # read stack of sinograms
	>>> data  = ntp.read_tiff_stack('./sinograms/img_0000.tiff')
	>>> _, na, nd  = data.shape
	>>> angles = np.linspace(0, np.pi*2, na, endpoint=False)
	>>> rec = ntp.reconstruct(data, angles, "FBP_CUDA", sinogram_order=True,
							  parameters={"FilterType":"hamming"})

	The filters for FBP_CUDA reconstruction included in the ASTRA toolbox are:

	 ``ram-lak`` (default), ``shepp-logan``, ``cosine``, ``hamming``, ``hann``, ``none``, ``tukey``,
	 ``lanczos``, ``triangular``, ``gaussian``, ``barlett-hann``, ``blackman``, ``nuttall``,
	 ``blackman-harris``, ``blackman-nuttall``, ``flat-top``, ``kaiser``,
	 ``parzen``, ``projection``, ``sinogram``, ``rprojection``, ``rsinogram``.

	GPU-based SIRT reconstruction with 100 iterations and limiting the values of
	the reconstructed pixel in the range [0,2]:

	>>> rec =  ntp.reconstruct(data, angles, "SIRT_CUDA", sinogram_order=True,
	parameters={"iterations":100, "MinConstraint": 0.0, "MaxConstraint" = 2.0 })
	"""
	if(tomo.ndim !=2 and tomo.ndim != 3 ):
		raise ValueError('Invalid shape of the array tomo. It must have 2 or 3 dimensions.')

	nd = tomo.shape[-1]
	pmat = get_astra_proj_matrix(nd, angles, method)

	if(tomo.ndim==2):
		out = recon_slice(tomo, method, pmat, parameters=parameters, pixel_size=pixel_size, offset=offset)
	elif(tomo.ndim==3):
		out = recon_stack(tomo, method, pmat, parameters=parameters, pixel_size=pixel_size, offset=offset, sinogram_order=sinogram_order)
	else:
		raise ValueError('Invalid array dimensions. Tomographic data must be 2D or 3D array.')

	return out
