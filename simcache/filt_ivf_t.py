import os, hashlib
import numpy  as np
import pickle as pk
import healpy as hp

import qcinv

import util

class library(object):
    # a collection of inverse-variance filtered maps.
    
    def __init__(self, sim_lib, lib_dir):
        self.sim_lib = sim_lib
        self.lib_dir = lib_dir

        if not os.path.exists(lib_dir):
            os.makedirs(lib_dir)

        if not os.path.exists(lib_dir + "/sim_hash.pk"):
            pk.dump( self.hashdict(), open(lib_dir + "/sim_hash.pk", 'w') )
        util.hash_check( pk.load( open(lib_dir + "/sim_hash.pk", 'r') ), self.hashdict() )

    def hashdict(self):
        # return a list of hashes used to
        # describe the parameters of this library,
        # used for sanity testing.
        return { 'sim_lib' : self.sim_lib.hashdict() }

    def get_dat_tlm(self, det):
        tfname = self.lib_dir + "/dat_det_" + det + "_tlm.npy"
        if not os.path.exists(tfname): self.cache_dat_t(det)
        return np.load(tfname)

    def get_sim_tlm(self, det, idx):
        tfname = self.lib_dir + "/sim_det_" + det + "_" + ('%04d' % idx) + "_tlm.npy"
        if not os.path.exists(tfname): self.cache_sim_t(det, idx)
        return np.load(tfname)

    def cache_sim_t(self, det, idx):
        tlm_fname = self.lib_dir + "/sim_det_" + det + "_" + ('%04d' % idx) + "_tlm.npy"

        assert( not os.path.exists(tlm_fname) )

        tlm = self.apply_ivf( det, self.sim_lib.get_sim_tmap(det, idx) )
        
        np.save(tlm_fname, tlm)

    def cache_dat_t(self, det):
        tlm_fname = self.lib_dir + "/dat_det_" + det + "_tlm.npy"

        assert( not os.path.exists(tlm_fname) )

        tlm = self.apply_ivf( det, self.sim_lib.get_dat_tmap(det) )
        
        np.save(tlm_fname, tlm)

    def apply_ivf( self, det, tmap ):
        assert(0)

    def get_fsky(self):
        mask_t = qcinv.util.load_map(self.mask_t)

        npix = len(mask_t)
        return mask_t.sum() / npix

class fl_library(library):
    # library for symmetric inverse-variance.
    
    def __init__(self, lmax, nside, bl, ftl, mask_t, sim_lib, lib_dir):
        # note: ftl, fel, fbl are the effective symmetric inverse-variance filter (after beam deconvolution).
        #       for a sky which has already been beam deconvolved by 1/(bl*pxw)

        self.lmax  = lmax
        self.nside = nside
        self.bl    = bl
        self.ftl   = ftl

        self.mask_t = mask_t
        
        super( fl_library, self ).__init__( sim_lib, lib_dir )

    def hashdict(self):
        # return a list of hashes used to
        # describe the parameters of this library,
        # used for sanity testing.
        return { 'lmax'    : self.lmax,
                 'nside'   : self.nside,
                 'bl'      : hashlib.sha1(self.bl.view(np.uint8)).hexdigest(),
                 'ftl'     : hashlib.sha1(self.ftl.view(np.uint8)).hexdigest(),
                 'mask_t'  : hashlib.sha1(qcinv.util.load_map(self.mask_t).view(np.uint8)).hexdigest(),
                 'sim_lib' : self.sim_lib.hashdict(), 
                 'super'   : super( fl_library, self ).hashdict() }

    def apply_ivf(self, det, tmap, pmap):
        mask_t = qcinv.util.load_map(self.mask_t)

        bl     = self.bl
        pxw    = hp.pixwin(self.nside)[0:self.lmax+1]

        tlm = hp.map2alm( tmap * mask_t, lmax=self.lmax, iter=0, regression=False )

        hp.almxfl( tlm, self.ftl / bl / pxw, inplace=True )
    
        return tlm

    def get_ftl(self, det):
        return self.ftl
