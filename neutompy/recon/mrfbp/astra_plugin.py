#-----------------------------------------------------------------------
#Copyright 2015 Daniel M. Pelt
#
#Contact: D.M.Pelt@cwi.nl
#Website: http://www.dmpelt.com
#
#
#This file is part of the PyMR-FBP, a Python implementation of the
#MR-FBP tomographic reconstruction method.
#
#PyMR-FBP is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#PyMR-FBP is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with PyMR-FBP. If not, see <http://www.gnu.org/licenses/>.
#
#-----------------------------------------------------------------------
import astra
import numpy as np
from six.moves import range
from scipy.signal import fftconvolve
import scipy.ndimage.filters as snf
import scipy.io as sio
import numpy.linalg as na

import os, errno

try:
    from pywt import wavedec2
except:
    astra.log.warn("No pywavelets installed, wavelet functions will not work.")

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise


class plugin(astra.plugin.ReconstructionAlgorithm2D):
    """Reconstructs using the MR-FBP method [1].

    Options:

    'save_filter' (optional): file to save filter to (overwrites existing)
    'use_saved_filter' (optional): file to load filter from
    'reg_grad' (optional): amount of l2 gradient minimization
    'reg_path' (optional): folder to save range of 'reg_grad' values to
    'reg_range' (optional): range of 'reg_grad' values to try (of form (min, max, number of trials))
    'reg_wav' (optional): amount of l2 wavelet minimization
    'wav_bas' (optional): wavelet to use (see pywt.wavelist())
    'nlinear' (optional): number of linear steps in exponential binning
    
    [1] Pelt, D. M., & Batenburg, K. J. (2014). Improving filtered backprojection 
        reconstruction by data-dependent filtering. Image Processing, IEEE 
        Transactions on, 23(11), 4750-4762.
    """
    
    astra_name="MR-FBP"
    
    def customFBP(self, f, s):
        sf = np.zeros_like(s)
        padded = np.zeros(s.shape[1]*2)
        l = int(s.shape[1]/2.)
        r = l+s.shape[1]
        bl = f.shape[0]/len(s)
        for i in range(sf.shape[0]):
            padded[l:r] = s[i]
            padded[:l] = padded[l]
            padded[r:] = padded[r-1]
            sf[i] = fftconvolve(padded,f,'same')[l:r]
        return (self.W.T*sf).reshape(self.v.shape)
    
    def initialize(self, cfg, nlinear=2, reg_wav=None, wav_bas='haar', reg_grad=None, save_filter=None, use_saved_filter=None, reg_path=None, reg_range=(1,100,10)):
        self.W = astra.OpTomo(cfg['ProjectorId'])
        self.reg_gr = reg_grad
        self.reg_wav = reg_wav
        self.save_filter=save_filter
        self.use_saved=use_saved_filter
        self.reg_path = reg_path
        self.reg_range = reg_range
        self.wav_bas = wav_bas
        
        if not self.use_saved:
            self.bck = (self.W.T*np.ones_like(self.s)<self.s.shape[0]-0.5).reshape(self.v.shape)
        
        if self.reg_path:
            self.reg_gr=1.
        
        fs = self.s.shape[1]
        if fs%2==0:
            fs += 1
        mf = int(fs/2)
        
        w=1
        c=mf
        
        bas = np.zeros(fs,dtype=np.float32)
        self.basis = []
        count=0
        while c<fs:
            bas[:]=0
            l = c
            r = c+w
            if r>fs: r=fs
            bas[l:r]=1
            if l!=0:
                l = fs-c-w
                r = l+w
                if l<0: l=0
                bas[l:r]=1
            self.basis.append(bas.copy())
            c += w
            count += 1
            if count>nlinear:
                w=2*w
        self.nf = len(self.basis)

    def run(self, iterations):
        if self.use_saved:
            flt = sio.loadmat(self.use_saved)['mrfbp_flt'].flatten()
            self.v[:] = self.customFBP(flt, self.s)
            return
        nrows = self.s.size
        ncols = self.nf
        if self.reg_gr:
            nrows += 2*self.v.size
        if self.reg_wav:
            q = wavedec2(self.v,self.wav_bas)
            l = [q[0].flatten()]
            for z in range(1,len(q)):
                l.extend([y.flatten() for y in q[z]])
            l = np.hstack(l)
            nrows += l.size
        A = np.zeros((nrows,ncols),dtype=np.float32)
        astra.log.info("Generating MR-FBP matrix")
        for i, bas in enumerate(self.basis):
            astra.log.info('{:.2f} % done'.format(100*float(i)/self.nf))
            img = self.customFBP(bas, self.s)
            if self.reg_wav:
                q = wavedec2(img,self.wav_bas)
                l = [q[0].flatten()]
                for z in range(1,len(q)):
                    l.extend([y.flatten() for y in q[z]])
                l = np.hstack(l)
                A[nrows-l.size:,i] = self.reg_wav*l
            if self.reg_gr!=None:
                dx = np.zeros_like(self.v)
                dx[:-1,:] = img[:-1,:] - img[1:,:]
                dy = np.zeros_like(self.v)
                dx[:,:-1] = img[:,:-1] - img[:,1:]
                dx[self.bck]=0
                dy[self.bck]=0
                img[self.bck]=0
                A[0:self.s.size,i] = self.W*img
                A[self.s.size+0*self.v.size:self.s.size+1*self.v.size,i] = self.reg_gr*dx.flatten()
                A[self.s.size+1*self.v.size:self.s.size+2*self.v.size,i] = self.reg_gr*dy.flatten()
            else:
                img[self.bck]=0
                astra.extrautils.clipCircle(img)
                A[:self.s.size,i] = self.W*img
        b = np.zeros(nrows,dtype=np.float32)
        b[:self.s.size] = self.s.flatten()
        if self.reg_path:
            import tifffile
            mkdir_p(self.reg_path)
            l, r, s = self.reg_range
            astra.log.info("Generating regularized images")
            for i, reg in enumerate(np.linspace(l,r,s)):
                astra.log.info('{:.2f} % done'.format(100*float(i)/s))
                A[self.s.size:nrows-self.v.size,:] *= reg
                out = na.lstsq(A,b, rcond=-1)
                flt = np.zeros_like(self.basis[0])
                for i, bas in enumerate(self.basis):
                    flt += out[0][i]*bas
                rc = self.customFBP(flt, self.s)
                tifffile.imsave(self.reg_path + os.sep + str(reg) + '.tiff', rc)
                A[self.s.size:nrows-self.v.size,:] /= reg
            return
        else:
            out = na.lstsq(A,b, rcond=-1)
        bl = self.basis[0].shape[0]
        flt = np.zeros_like(self.basis[0])
        for i, bas in enumerate(self.basis):
            flt += out[0][i]*bas
        if self.save_filter:
            sio.savemat(self.save_filter, {'mrfbp_flt':flt}, do_compression=True, appendmat=False)
        astra.log.info("Done!")
        rc = self.customFBP(flt, self.s)
        self.v[:] = rc
