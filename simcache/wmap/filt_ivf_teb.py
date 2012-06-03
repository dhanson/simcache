import os, hashlib
import numpy  as np
import pickle as pk
import healpy as hp

import qcinv
import simcache

# --

import dmc

def get_ftebl_eff( lmax, year, nside, det, forered, cl, mask_t, mask_p ):
    noiseT_uK_arcmin, noiseP_uK_arcmin = dmc.get_nlev_tp_uK_arcmin(year, det, forered, mask_t, mask_p)

    bl  = dmc.get_bl( year, det)[0:lmax+1]
    pxw = hp.pixwin( nside )[0:lmax+1]

    ftl = 1.0 / (cl.cltt[0:lmax+1] + (noiseT_uK_arcmin * np.pi/180./60.)**2 / (bl * pxw)**2); ftl[0:2] = 0.0
    fel = 1.0 / (cl.clee[0:lmax+1] + (noiseP_uK_arcmin * np.pi/180./60.)**2 / (bl * pxw)**2); fel[0:2] = 0.0
    fbl = 1.0 / (cl.clbb[0:lmax+1] + (noiseP_uK_arcmin * np.pi/180./60.)**2 / (bl * pxw)**2); fbl[0:2] = 0.0

    return ftl, fel, fbl

class fl_library(simcache.filt_ivf_teb.library):
    # library for symmetric inverse-variance.
    
    def __init__(self, lmax, cl, mask_t, mask_p, sim_lib, lib_dir):
        # note: ftl, fel, fbl are the effective symmetric inverse-variance filter,
        #       for a sky which has already been beam deconvolved by 1/(bl*pxw)

        self.lmax   = lmax
        self.cl     = cl
        self.mask_t = mask_t
        self.mask_p = mask_p

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
                 'mask_p'  : hashlib.sha1(qcinv.util.load_map(self.mask_p).view(np.uint8)).hexdigest(),
                 'sim_lib' : self.sim_lib.hashdict(), 
                 'super'   : super( fl_library, self ).hashdict() }

    def apply_ivf(self, det, tmap, pmap):
        mask_t = qcinv.util.load_map(self.mask_t)
        mask_p = qcinv.util.load_map(self.mask_p)

        tlm = hp.map2alm( tmap * mask_t, lmax=self.lmax, iter=0, regression=False )
        elm, blm = hp.map2alm_spin( (pmap.real * mask_p, pmap.imag * mask_p), 2, lmax=self.lmax )

        bl  = dmc.get_bl(self.sim_lib.year, det)[0:self.lmax+1]
        pxw = hp.pixwin( self.sim_lib.nside )[0:self.lmax+1]

        ftl, fel, fbl = self.get_ftebl(det)

        hp.almxfl( tlm, ftl / bl / pxw, inplace=True )
        hp.almxfl( elm, fel / bl / pxw, inplace=True )
        hp.almxfl( blm, fbl / bl / pxw, inplace=True )

        return tlm, elm, blm

    def get_ftebl(self, det):
        return get_ftebl_eff( self.lmax, self.sim_lib.year, self.sim_lib.nside, det, self.sim_lib.forered,
                              self.cl, qcinv.util.load_map(self.mask_t), qcinv.util.load_map(self.mask_p) )


class qcinv_t_library(simcache.filt_ivf_teb.library):
    # full c^{-1} filtering for T, diagonal for E and B.

    def __init__(self, lmax, cl, mask_t, mask_p, sim_lib, lib_dir):
        self.lmax   = lmax
        self.cl     = cl
        self.mask_t = mask_t
        self.mask_p = mask_p

        super( qcinv_t_library, self ).__init__( sim_lib, lib_dir )

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
                 'mask_p'  : hashlib.sha1(qcinv.util.load_map(self.mask_p).view(np.uint8)).hexdigest(),
                 'sim_lib' : self.sim_lib.hashdict(),
                 'super'   : super( qcinv_t_library, self ).hashdict() }

    def apply_ivf(self, det, tmap, pmap):
        assert(self.lmax == 1000)

        mask_t = qcinv.util.load_map(self.mask_t)
        mask_p = qcinv.util.load_map(self.mask_p)

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

        # simple filtering for polarization.
        elm, blm = hp.map2alm_spin( (pmap.real * mask_p, pmap.imag * mask_p), 2, lmax=self.lmax )
        ftl, fel, fbl = self.get_ftebl(det)
        hp.almxfl( elm, fel / bl / pxw, inplace=True )
        hp.almxfl( blm, fbl / bl / pxw, inplace=True )

        return tlm, elm, blm

    def get_ftebl(self, det):
        return get_ftebl_eff( self.lmax, self.sim_lib.year, self.sim_lib.nside, det, self.sim_lib.forered,
                              self.cl, qcinv.util.load_map(self.mask_t), qcinv.util.load_map(self.mask_p) )

class qcinv_library(simcache.filt_ivf_teb.library):
    # library for symmetric inverse-variance.

    def __init__(self, lmax, cl, mask_t, mask_p, sim_lib, lib_dir, diag_n=False):
        self.lmax   = lmax
        self.cl     = cl
        self.mask_t = mask_t
        self.mask_p = mask_p
        self.diag_n = diag_n

        super( qcinv_library, self ).__init__( sim_lib, lib_dir )

    def hashdict(self):
        # return a list of hashes used to
        # describe the parameters of this library,
        # used for sanity testing.
        slinv = qcinv.opfilt_tp.alm_filter_sinv(self.cl)

        return { 'lmax'    : self.lmax,
                 'slinv'   : slinv.hashdict(),
                 'mask_t'  : hashlib.sha1(qcinv.util.load_map(self.mask_t).view(np.uint8)).hexdigest(),
                 'mask_p'  : hashlib.sha1(qcinv.util.load_map(self.mask_p).view(np.uint8)).hexdigest(),
                 'diag_n'  : self.diag_n,
                 'sim_lib' : self.sim_lib.hashdict(),
                 'super'   : super( qcinv_library, self ).hashdict() }

    def apply_ivf(self, det, tmap, pmap):
        assert(self.lmax == 1000)

        mask_t = qcinv.util.load_map(self.mask_t)
        mask_p = qcinv.util.load_map(self.mask_p)

        bl  = dmc.get_bl(self.sim_lib.year, det)[0:self.lmax+1]
        pxw = hp.pixwin( self.sim_lib.nside )[0:self.lmax+1]

        # qcinv filtering for temperature
        dcf = self.lib_dir + "/dense_cache_det_" + det + ".pk"
        #                  id         preconditioners                 lmax    nside     im      em            tr                      cache
        chain_descr = [ [  2, ["split(dense("+dcf+"), 32, diag_cl)"],  256,   128,       3,     0.0,  qcinv.cd_solve.tr_cg,  qcinv.cd_solve.cache_mem()],
                        [  1, ["split(stage(2), 256, diag_cl)"],       512,   256,       3,     0.0,  qcinv.cd_solve.tr_cg,  qcinv.cd_solve.cache_mem()],
                        [  0, ["split(stage(1), 512, diag_cl)"],      1000,   512,  np.inf,  1.0e-6,  qcinv.cd_solve.tr_cg,  qcinv.cd_solve.cache_mem()] ]

        # temperature noise covariance
        ninv_t = ( hp.read_map( dmc.get_fname_iqumap(self.sim_lib.year, det, self.sim_lib.forered), hdu=1, field=3 ) /
                   dmc.sigma0[(self.sim_lib.year, self.sim_lib.forered, 'T')][det]**2 / 1e6 * mask_t )

        # polarization noise covariance
        tfname =  dmc.get_fname_iqumap(self.sim_lib.year, det, False) #NOTE: Nobs counts taken from non-FG reduced maps.

        n_inv_qq = hp.read_map( tfname, hdu=2, field=1 ) / (dmc.sigma0[(self.sim_lib.year, self.sim_lib.forered, 'P')][det] * 1.e3)**2 * mask_p
        n_inv_qu = hp.read_map( tfname, hdu=2, field=2 ) / (dmc.sigma0[(self.sim_lib.year, self.sim_lib.forered, 'P')][det] * 1.e3)**2 * mask_p
        n_inv_uu = hp.read_map( tfname, hdu=2, field=3 ) / (dmc.sigma0[(self.sim_lib.year, self.sim_lib.forered, 'P')][det] * 1.e3)**2 * mask_p

        # construct filter
        if self.diag_n == True:
            n_inv_filt = qcinv.opfilt_tp.alm_filter_ninv( [ninv_t, 0.5*(n_inv_qq + n_inv_uu)], bl*pxw, marge_monopole=True, marge_dipole=True )
        else:
            n_inv_filt = qcinv.opfilt_tp.alm_filter_ninv( [ninv_t, n_inv_qq, n_inv_qu, n_inv_uu], bl*pxw, marge_monopole=True, marge_dipole=True )

        chain = qcinv.multigrid.multigrid_chain( qcinv.opfilt_tp, chain_descr, self.cl, n_inv_filt )

        alm = qcinv.opfilt_tp.teblm( [np.zeros( qcinv.util_alm.lmax2nlm(self.lmax), dtype=np.complex ) for i in xrange(0,3)] )
        chain.solve( alm, [tmap, pmap.real, pmap.imag] )

        return alm.tlm, alm.elm, alm.blm

    def get_ftebl(self, det):
        return get_ftebl_eff( self.lmax, self.sim_lib.year, self.sim_lib.nside, det, self.sim_lib.forered,
                              self.cl, qcinv.util.load_map(self.mask_t), qcinv.util.load_map(self.mask_p) )
