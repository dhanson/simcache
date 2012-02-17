import os, hashlib, math
import pickle as pk
import numpy  as np
import healpy as hp

import simcache

# --

import dmc

class cmb_nse_library(simcache.sims_tmap.library):
    # a library of noise realizations generated from a map of N^{-1}
    # on top of CMB realizations retrieved from a sim_teblm_cmb,
    # convolved with a symmetric beam.
    #

    def __init__(self, lmax, nside, year, sim_tlm_cmb, forered, ntype, lib_dir):
        assert( lmax  == 1000 )
        assert( nside == 512 )
        
        self.lmax           = lmax
        self.nside          = nside
        
        self.year           = year
        self.sim_tlm_cmb    = sim_tlm_cmb
        self.forered        = forered
        self.ntype          = ntype
    
        super( cmb_nse_library, self ).__init__( lib_dir=lib_dir )

    def hashdict(self):
        return { 'lmax'          : self.lmax,
                 'nside'         : self.nside,
                 'year'          : self.year,
                 'sim_tlm_cmb'   : self.sim_tlm_cmb.hashdict(),
                 'forered'       : self.forered,
                 'nytpe'         : self.ntype,
                 'lmax'          : self.lmax,
                 'nside'         : self.nside,
                 'super'         : super( cmb_nse_library, self ).hashdict() }

    def get_dat_tmap(self, det):
        return hp.read_map( dmc.get_fname_iqumap(self.year, det, self.forered), hdu=1, field=0 ) * 1.e3

    def get_beam(self, det):
        pxw  = hp.pixwin(self.nside)[0:self.lmax+1]
        beam = dmc.get_bl(self.year, det)[0:self.lmax+1]

        return beam * pxw

    def simulate(self, det, idx):
        assert( det in (dmc.wmap_das + dmc.wmap_bands) )
        
        tlm = self.sim_tlm_cmb.get_sim_tlm(idx)
        
        hp.almxfl(tlm, self.get_beam(det), inplace=True)
            
        tmap = hp.alm2map(tlm, self.nside)

        tmap_nse = self.simulate_nse(det)
        tmap += tmap_nse; del tmap_nse
        
        return tmap

    def simulate_nse(self, det):
        print 'simulate noise forered = ', self.forered
        if self.ntype == 'nobs':
            rtnobsinv = 1.0 / np.sqrt( hp.read_map( dmc.get_fname_iqumap(self.year, det, self.forered), hdu=1, field=3 ) )
            npix = 12*self.nside**2
            
            t_nse = rtnobsinv * dmc.sigma0[(self.year, self.forered, 'T')][det] * 1e3 * np.random.standard_normal( npix )
        else:
            assert(0)

        return t_nse

