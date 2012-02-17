import os, hashlib, math
import pickle as pk
import numpy  as np
import healpy as hp

import util
import sims_teblm

class library(object):
    # library of sims.

    def __init__(self, lib_dir):
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
        return {}

    def get_sim_tmap(self, det, idx):
        tfname = self.lib_dir + "/sim_det_" + det + "_" + ('%04d' % idx) + "_tmap.npy"
        if not os.path.exists(tfname): self.cache_sim_tp(det, idx)
        return np.load(tfname)

    def get_sim_pmap(self, det, idx):
        tfname = self.lib_dir + "/sim_det_" + det + "_" + ('%04d' % idx) + "_pmap.npy"
        if not os.path.exists(tfname): self.cache_tp(idx)
        return np.load(tfname)
    
    def cache_sim_tp(self, det, idx):
        tmap_fname = self.lib_dir + "/sim_det_" + det + "_" + ('%04d' % idx) + "_tmap.npy"
        pmap_fname = self.lib_dir + "/sim_det_" + det + "_" + ('%04d' % idx) + "_pmap.npy"

        assert( not any(os.path.exists(fname) for fname in [tmap_fname, pmap_fname] ) )
            
        tmap, pmap = self.simulate(det, idx)
        
        np.save(tmap_fname, tmap)
        np.save(pmap_fname, pmap)
    
    def simulate(self, det, idx):
        assert(0)


class cmb_nse_homog_library(library):
    def __init__(self, lmax, nside, bl, noiseT_uK_arcmin, noiseP_uK_arcmin, sim_teblm_cmb, lib_dir):
        self.lmax             = lmax
        self.nside            = nside
        self.bl               = bl
        self.noiseT_uK_arcmin = noiseT_uK_arcmin
        self.noiseP_uK_arcmin = noiseP_uK_arcmin
        self.sim_teblm_cmb    = sim_teblm_cmb

        super( cmb_nse_homog_library, self ).__init__( lib_dir=lib_dir )

    def hashdict(self):
        return { 'lmax'             : self.lmax,
                 'nside'            : self.nside,
                 'bl'               : hashlib.sha1(self.bl.view(np.uint8)).hexdigest(),
                 'noiseT_uK_arcmin' : self.noiseT_uK_arcmin,
                 'noiseP_uK_arcmin' : self.noiseP_uK_arcmin,
                 'sim_teblm_cmb'    : self.sim_teblm_cmb.hashdict(),
                 'lib_dir'          : self.lib_dir,
                 'super'            : super( cmb_nse_homog_library, self ).hashdict() }

    def get_dat_tmap(self, det):
        return self.get_sim_tmap(det, -1)

    def get_dat_pmap(self, det):
        return self.get_sim_pmap(det, -1)

    def simulate(self, det, idx):
        assert(det == '')
        
        tlm = self.sim_teblm_cmb.get_sim_tlm(idx)
        elm = self.sim_teblm_cmb.get_sim_elm(idx)
        blm = self.sim_teblm_cmb.get_sim_blm(idx)

        beam = self.bl[0:self.lmax+1] * hp.pixwin(self.nside)[0:self.lmax+1]
        hp.almxfl(tlm, beam, inplace=True)
        hp.almxfl(elm, beam, inplace=True)
        hp.almxfl(blm, beam, inplace=True)
        
        tmap = hp.alm2map(tlm, self.nside)
        qmap, umap = hp.alm2map_spin( (elm, blm), self.nside, 2, lmax=self.lmax )

        npix = 12*self.nside**2
        tmap += np.random.standard_normal(npix) * (self.noiseT_uK_arcmin * np.sqrt(npix / 4. / np.pi) * np.pi / 180. / 60.)
        qmap += np.random.standard_normal(npix) * (self.noiseP_uK_arcmin * np.sqrt(npix / 4. / np.pi) * np.pi / 180. / 60.)
        umap += np.random.standard_normal(npix) * (self.noiseP_uK_arcmin * np.sqrt(npix / 4. / np.pi) * np.pi / 180. / 60.)

        return tmap, qmap + 1.j*umap
