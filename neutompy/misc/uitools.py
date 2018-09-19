from tkinter import *
from tkinter import filedialog
import os
import logging
import tkinter

__author__  = "Davide Micieli"
__all__     = ['get_image_gui',
	       'save_filename_gui',
		'get_folder_gui',
		'get_filename_gui',
		'get_screen_resolution']

#logging.basicConfig(level=logging.WARNING)
logs = logging.getLogger(__name__)


def get_image_gui(initialdir='', message='Select image...'):
	"""
	This function opens a dialog box to select an image (TIFF or FITS) and to get
	its file path.

	Parameters
	----------
	initialdir : str, optional
		String defining the path of the initial folder to open in the dialog box.

	message : str, optional
		String defining the dialog box title.

	Returns
	-------
	fname : str
		String defining the file path selected using the dialog box.
	"""

	if not (initialdir):
		initialdir = os.path.abspath(os.sep)

	root = Tk()
	root.withdraw()
	root.wm_attributes('-topmost', 1)

	while True:
		fname = filedialog.askopenfilename(initialdir=initialdir, title=message,
						filetypes = (("Image files", "*.tif *.tiff *.fits"),
						("TIFF files","*.tif *.tiff"),("FITS files","*.fits")))

		if(fname):
			break

	return fname


def save_filename_gui(initialdir='', message='Select folder and the name of the file to save...' ):
	"""
	This function opens a dialog box to select a file to save and get its file path.

	Parameters
	----------
	initialdir : str, optional
		String defining the path of the initial folder to open in the dialog box.

	message : str, optional
		String defining the dialog box title.

	Returns
	-------
	fname : str
		String defining the file path selected using the dialog box.
	"""

	if not (initialdir):
		initialdir = os.path.abspath(os.sep)

	root = Tk()
	root.withdraw()
	root.wm_attributes('-topmost', 1)

	while True:
		fname =  filedialog.asksaveasfilename(initialdir = initialdir, title = message)

		if(fname):
			break


	return fname


def get_folder_gui(initialdir='', message='Select folder...'):
	"""
	This function opens a dialog box to select a folder and get its file path.

	Parameters
	----------
	initialdir : str, optional
		String defining the path of the initial folder to open in the dialog box.

	message : str, optional
		String defining the dialog box title.

	Returns
	-------
	fname : str
		String defining the folder path selected using the dialog box.
	"""

	if not (initialdir):
		initialdir = os.path.abspath(os.sep)

	root = Tk()
	root.withdraw()
	root.wm_attributes('-topmost', 1)


	while True:
		fname = filedialog.askdirectory(initialdir=initialdir, title=message)

		if(fname):
			break

	return fname


def get_filename_gui(initialdir='', message='Select file...', ext=None ):
	"""
	This function opens a dialog box to select a file and get its file path.

	Parameters
	----------
	initialdir : str, optional
		String defining the path of the initial folder to open in the dialog box.

	message : str, optional
		String defining the dialog box title.

	ext : tuple, optional
		Tuple defining the file types to show. It includes the description and a
		shell-style wildcards defining the extension of the files.
		E.g. to filter TIFF images: (('Tiff iamges', '*.tiff *.tif'))

	Returns
	-------
	fname : str
		String defining the file path selected using the dialog box.
	"""

	if not (initialdir):
		initialdir = os.path.abspath(os.sep)

	root = Tk()
	root.withdraw()
	root.wm_attributes('-topmost', 1)

	while True:

		if ext is None:
			fname = filedialog.askopenfilename(initialdir=initialdir, title=message)
		else:
			ext = ( ext, ("All files", "*.*") )
			fname = filedialog.askopenfilename(initialdir=initialdir, title=message,
			 				   filetypes = ext)

		if(fname):
			break

	return fname


def get_screen_resolution():
	"""
	This function returns the screen resolution as tuple.

	Example
	-------
	>>> width, height = ntp.get_screen_resolution()
	"""
	root = tkinter.Tk()
	root.withdraw()
	width = root.winfo_screenwidth()
	height = root.winfo_screenheight()
	return (width, height)
