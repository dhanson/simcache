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


        ninv = ( hp.read_map( dmc.get_fname_imap(self.sim_lib.year, det, self.sim_lib.forered), hdu=1, field=1 ) /
                 dmc.sigma0[(self.sim_lib.year, self.sim_lib.forered, 'T')][det]**2 / 1e6 * mask_t ) 
        n_inv_filt = qcinv.opfilt_tt.alm_filter_ninv( ninv, bl*pxw, marge_monopole=True, marge_dipole=True, marge_maps=[] )
        chain = qcinv.multigrid.multigrid_chain( qcinv.opfilt_tt, chain_descr, self.cl, n_inv_filt )

        tlm = np.zeros( qcinv.util_alm.lmax2nlm(self.lmax), dtype=np.complex )
        chain.solve( tlm, tmap )

        return tlm

    def get_ftl(self, det):
        return get_ftl_eff( self.lmax, self.sim_lib.year, self.sim_lib.nside, det, self.sim_lib.forered,
                            self.cl, qcinv.util.load_map(self.mask_t) )

class qcinv_multi_simple_library(simcache.filt_ivf_t.library):
    # library for symmetric inverse-variance.

    def __init__(self, lmax, cl, mask_t, sim_lib, lib_dir):
        self.lmax   = lmax
        self.cl     = cl
        self.mask_t = mask_t

        self.sim_lib = sim_lib
        self.lib_dir = lib_dir

        if not os.path.exists(lib_dir):
            os.makedirs(lib_dir)

        if not os.path.exists(lib_dir + "/sim_hash.pk"):
            pk.dump( self.hashdict(), open(lib_dir + "/sim_hash.pk", 'w') )
        simcache.util.hash_check( pk.load( open(lib_dir + "/sim_hash.pk", 'r') ), self.hashdict() )

    def detstr2dets(self, det):
        dets = []
        if det == 'QVW':
            return ['Q', 'V', 'W']
        else:
            assert(0)

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
                 'sim_lib' : self.sim_lib.hashdict() }

    def cache_sim_t(self, detstr, idx):
        dets = self.detstr2dets(detstr)

        tlm_fname = self.lib_dir + "/sim_det_" + detstr + "_" + ('%04d' % idx) + "_tlm.npy"

        assert( not os.path.exists(tlm_fname) )

        tlm = self.apply_ivf( dets, [self.sim_lib.get_sim_tmap(det, idx) for det in dets] )

        np.save(tlm_fname, tlm)

    def cache_dat_t(self, detstr):
        dets = self.detstr2dets(detstr)

        tlm_fname = self.lib_dir + "/dat_det_" + detstr + "_tlm.npy"

        assert( not os.path.exists(tlm_fname) )

        tlm = self.apply_ivf( dets, [self.sim_lib.get_dat_tmap(det) for det in dets] )

        np.save(tlm_fname, tlm)

    def get_fsky(self):
        mask_t = qcinv.util.load_map(self.mask_t)

        npix = len(mask_t)
        return mask_t.sum() / npix

    def apply_ivf(self, dets, tmaps):
        assert(self.lmax == 1000)

        n_inv_filts = []
        for det in dets:
            mask_t = qcinv.util.load_map(self.mask_t)

            bl  = dmc.get_bl(self.sim_lib.year, det)[0:self.lmax+1]
            pxw = hp.pixwin( self.sim_lib.nside )[0:self.lmax+1]

            # qcinv filtering for temperature
            dcf = self.lib_dir + "/dense_cache_det_" + det + ".pk"
            #                  id         preconditioners                 lmax    nside     im      em            tr                      cache
            chain_descr = [ [  2, ["split(dense("+dcf+"), 64, diag_cl)"],  256,   128,       3,     0.0,  qcinv.cd_solve.tr_cg,  qcinv.cd_solve.cache_mem()],
                            [  1, ["split(stage(2), 256, diag_cl)"],       512,   256,       3,     0.0,  qcinv.cd_solve.tr_cg,  qcinv.cd_solve.cache_mem()],
                            [  0, ["split(stage(1), 512, diag_cl)"],      1000,   512,  np.inf,  1.0e-6,  qcinv.cd_solve.tr_cg,  qcinv.cd_solve.cache_mem()] ]

            ninv = ( hp.read_map( dmc.get_fname_imap(self.sim_lib.year, det, self.sim_lib.forered), hdu=1, field=1 ) /
                     dmc.sigma0[(self.sim_lib.year, self.sim_lib.forered, 'T')][det]**2 / 1e6 * mask_t )
            n_inv_filt = qcinv.opfilt_tt.alm_filter_ninv( ninv, bl*pxw, marge_monopole=True, marge_dipole=True, marge_maps=[] )
            n_inv_filts.append( n_inv_filt )
        n_inv_filts = qcinv.opfilt_tt_multi_simple.alm_filter_ninv_filts( n_inv_filts, degrade_single=True )

        chain = qcinv.multigrid.multigrid_chain( qcinv.opfilt_tt_multi_simple, chain_descr, self.cl, n_inv_filts )

        tlm = np.zeros( qcinv.util_alm.lmax2nlm(self.lmax), dtype=np.complex )
        chain.solve( tlm, tmaps )

        return tlm

    def get_ftl(self, detstr):
        dets = self.detstr2dets(detstr)

        clnn    = np.zeros(self.lmax+1)
        for det in dets:
            noiseT_uK_arcmin = dmc.get_nlev_t_uK_arcmin(self.sim_lib.year, det, self.sim_lib.forered, qcinv.util.load_map(self.mask_t))

            bl  = dmc.get_bl( self.sim_lib.year, det)[0:self.lmax+1]
            pxw = hp.pixwin( self.sim_lib.nside )[0:self.lmax+1]

            clnn += (bl * pxw)**2 / (noiseT_uK_arcmin * np.pi/180./60.)**2
        clnn = 1./clnn

        ftl = 1.0 / (self.cl.cltt[0:self.lmax+1] + clnn); ftl[0:2] = 0.0

        return ftl
