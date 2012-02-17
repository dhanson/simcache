import os, hashlib
import numpy  as np
import pickle as pk
import healpy as hp

import qcinv
import simcache

# --

import dmc

def get_ftl_eff( lmax, year, nside, det, forered, cl, mask_t ):
    noiseT_uK_arcmin = dmc.get_nlev_t_uK_arcmin(year, det, forered, mask_t)

    bl  = dmc.get_bl( year, det)[0:lmax+1]
    pxw = hp.pixwin( nside )[0:lmax+1]

    ftl = 1.0 / (cl.cltt[0:lmax+1] + (noiseT_uK_arcmin * np.pi/180./60.)**2 / (bl * pxw)**2); ftl[0:2] = 0.0

    return ftl

class fl_library(simcache.filt_ivf_t.library):
    # library for symmetric inverse-variance.
    
    def __init__(self, lmax, cl, mask_t, sim_lib, lib_dir):
        # note: ftl, fel, fbl are the effective symmetric inverse-variance filter,
        #       for a sky which has already been beam deconvolved by 1/(bl*pxw)

        self.lmax   = lmax
        self.cl     = cl
        self.mask_t = mask_t

        super( fl_library, self ).__init__( sim_lib, lib_dir )

    def hashdict(self):
        # return a list of hashes used to
        # describe the parameters of this library,
        # used for sanity testing.
        return { 'lmax'    : self.lmax,
                 'cltt'    : hashlib.sha1(self.cl.cltt.view(np.uint8)).hexdigest(),
                 'clte'    : hashlib.sha1(self.cl.clte.view(np.uint8)).hexdigest(),
                 'clee'    : hashlib.sha1(self.cl.clee.view(np.uint8)).hexdigest(),
                 'clbb'    : hashlib.sha1(self.cl.clbb.view(np.uint8)).hexdigest(),
                 'mask_t'  : hashlib.sha1(qcinv.util.load_map(self.mask_t).view(np.uint8)).hexdigest(),
                 'sim_lib' : self.sim_lib.hashdict(), 
                 'super'   : super( fl_library, self ).hashdict() }

    def apply_ivf(self, det, tmap):
        mask_t = qcinv.util.load_map(self.mask_t)

        tlm = hp.map2alm( tmap * mask_t, lmax=self.lmax, iter=0, regression=False )

        bl  = dmc.get_bl(self.sim_lib.year, det)[0:self.lmax+1]
        pxw = hp.pixwin( self.sim_lib.nside )[0:self.lmax+1]

        ftl = self.get_ftl(det)

        hp.almxfl( tlm, ftl / bl / pxw, inplace=True )

        return tlm

    def get_ftl(self, det):
        return get_ftl_eff( self.lmax, self.sim_lib.year, self.sim_lib.nside, det, self.sim_lib.forered,
                            self.cl, qcinv.util.load_map(self.mask_t) )


class qcinv_library(simcache.filt_ivf_t.library):
    # library for symmetric inverse-variance.

    def __init__(self, lmax, cl, mask_t, sim_lib, lib_dir):
        self.lmax   = lmax
        self.cl     = cl
        self.mask_t = mask_t

        super( qcinv_library, self ).__init__( sim_lib, lib_dir )

    def hashdict(self):
        # return a list of hashes used to
        # describe the parameters of this library,
        # used for sanity testing.
        return { 'lmax'    : self.lmax,
                 'cltt'    : hashlib.sha1(self.cl.cltt.view(np.uint8)).hexdigest(),
                 'clte'    : hashlib.sha1(self.cl.clte.view(np.uint8)).hexdigest(),
                 'clee'    : hashlib.sha1(self.cl.clee.view(np.uint8)).hexdigest(),
                 'clbb'    : hashlib.sha1(self.cl.clbb.view(np.uint8)).hexdigest(),
                 'mask_t'  : hashlib.sha1(qcinv.util.load_map(self.mask_t).view(np.uint8)).hexdigest(),
                 'sim_lib' : self.sim_lib.hashdict(),
                 'super'   : super( qcinv_library, self ).hashdict() }

    def apply_ivf(self, det, tmap):
        assert(self.lmax == 1000)

        mask_t = qcinv.util.load_map(self.mask_t)

        bl  = dmc.get_bl(self.sim_lib.year, det)[0:self.lmax+1]
        pxw = hp.pixwin( self.sim_lib.nside )[0:self.lmax+1]

        # qcinv filtering for temperature
        dcf = self.lib_dir + "/dense_cache_det_" + det + ".pk"
        #                  id         preconditioners                 lmax    nside     im      em            tr                      cache
        chain_descr = [ [  2, ["split(dense("+dcf+"), 64, diag_cl)"],  256,   128,       3,     0.0,  qcinv.cd_solve.tr_cg,  qcinv.cd_solve.cache_mem()],
                        [  1, ["split(stage(2), 256, diag_cl)"],       512,   256,       3,     0.0,  qcinv.cd_solve.tr_cg,  qcinv.cd_solve.cache_mem()],
                        [  0, ["split(stage(1), 512, diag_cl)"],      1000,   512,  np.inf,  1.0e-6,  qcinv.cd_solve.tr_cg,  qcinv.cd_solve.cache_mem()] ]


        ninv = ( hp.read_map( dmc.get_fname_iqumap(self.sim_lib.year, det, self.sim_lib.forered), hdu=1, field=3 ) /
                 dmc.sigma0[(self.sim_lib.year, self.sim_lib.forered, 'T')][det]**2 / 1e6 * mask_t ) 
        n_inv_filt = qcinv.opfilt_tt.alm_filter_ninv( ninv, bl*pxw, marge_monopole=True, marge_dipole=True, marge_maps=[] )
        chain = qcinv.multigrid.multigrid_chain( qcinv.opfilt_tt, chain_descr, self.cl, n_inv_filt )

        tlm = np.zeros( qcinv.util_alm.lmax2nlm(self.lmax), dtype=np.complex )
        chain.solve( tlm, tmap )

        return tlm

    def get_ftl(self, det):
        return get_ftl_eff( self.lmax, self.sim_lib.year, self.sim_lib.nside, det, self.sim_lib.forered,
                            self.cl, qcinv.util.load_map(self.mask_t) )
