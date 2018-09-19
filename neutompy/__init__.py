#__version__ = '1.0.0'
#execfile('version.py')


#version = {}
#with open("neutompy/version.py") as fp:
#    exec(fp.read(), version)
#__version__ = version['__version__']

import os

# load __version__ variable
package_root_dir = os.path.dirname(os.path.abspath(__file__))
file_version     = os.path.join(package_root_dir, "version.py")
exec(open(file_version).read())


from .image.image import *
from .image.rebin import *
from .preproc.preproc import *
from .recon.recon import *
from .misc.uitools import *
from .postproc.convert import *
from .postproc.crop import *
from .metrics.metrics import *
