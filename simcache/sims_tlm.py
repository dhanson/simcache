import os, hashlib, math
import pickle as pk
import numpy  as np
import healpy as hp

import util

class library(object):
    # library of temperature alms.

    def __init__(self, lib_dir):
        assert(lib_dir != None)
        
        self.lib_dir = lib_dir

        if not os.path.exists(lib_dir):
            os.makedirs(lib_dir)

        if not  os.path.exists(lib_dir + "/sim_hash.pk"):
            pk.dump( self.hashdict(), open(lib_dir + "/sim_hash.pk", 'w') )
        util.hash_check( pk.load( open(lib_dir + "/sim_hash.pk", 'r') ), self.hashdict() )

    def hashdict(self):
        # return a list of hashes used to
        # describe the parameters of this library,
        # used for sanity testing.
        return {}

    def get_sim_tlm(self, idx):
        tfname = self.lib_dir + "/sim_" + ('%04d' % idx) + "_tlm.npy"

        if not os.path.exists(tfname):
            self.cache_t(idx)

        return np.load(tfname)

    def cache_t(self, idx):
        tlm_fname = self.lib_dir + "/sim_" + ('%04d' % idx) + "_tlm.npy"
        assert( not os.path.exists(tlm_fname) )

        tlm = self.simulate(idx)

        np.save(tlm_fname, tlm)

    def simulate(self, idx):
        assert(0)

class cmb_library(library):
    # library of temperature alms, with or without lensing.

    def __init__(self, lmax, cl, lib_dir):
        self.lmax = lmax
        self.cl   = cl

        super( cmb_library, self ).__init__( lib_dir=lib_dir )

    def hashdict(self):
        return { 'lmax'    : self.lmax,
                 'cltt'    : hashlib.sha1(self.cl.cltt.view(np.uint8)).hexdigest(),
                 'lib_dir' : self.lib_dir,
                 'super'   : super( cmb_library, self ).hashdict() }

    def simulate(self, idx):
        tlm = hp.synalm( self.cl.cltt, lmax=self.lmax )
        return tlm
