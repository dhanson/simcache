#!/usr/bin/env python

import os, imp, sys, time, getopt
import numpy as np
import pylab as pl

import qcinv
import simcache

import simcache.sims_tlm        as sims_tlm
import simcache.wmap.sims_tmap  as sims_tmap
import simcache.wmap.filt_ivf_t as filt_ivf_t

# --

prefix  = "sims_t_qv"
dets    = ['Q', 'V']

year    = 7
forered = True
ntype   = 'nobs'
lmax    = 1000
nside   = 512

nsims   = 1

mask_t  = simcache.wmap.dmc.get_fname_temperature_analysis_mask(year, nside)

cl = simcache.wmap.dmc.get_bestfit_cl_lensed(7, 'lcdm_sz_lens')

# sim params.
sims_cmb = sims_tlm.cmb_library(lmax, cl, "scratch/tlm_cmb_library_wmap")
sims     = sims_tmap.cmb_nse_library(lmax, nside, year, sims_cmb, forered, ntype=ntype, lib_dir=("scratch/" + prefix + "/sims"))

# ivf params.
ivfs_qc  = filt_ivf_t.qcinv_multi_simple_library(dets, lmax, cl, mask_t, sims, lib_dir=("scratch/" + prefix + "/ivfs_qc"))

# -- spectra
if not os.path.exists("scratch/" + prefix): os.makedirs("scratch/" + prefix)

cltt = cl.cltt[0:lmax+1]

# -- mask stats
fsky_tt = ivfs_qc.get_fsky()
# --

t0 = time.time()

# -- load data
cltt_bar_dat_qc = qcinv.util_alm.alm_cl( ivfs_qc.get_dat_tlm() )

# -- load sims
cltt_bar_sim_avg_qc = simcache.util.avg()
for i in xrange(0, nsims):
    print 'loading sim i = ', i, ' elapsed =  %0.3f s' % (time.time()-t0)
    cltt_bar_sim_avg_qc += qcinv.util_alm.alm_cl( ivfs_qc.get_sim_tlm(i) )

# -- make plots
pl.figure(figsize=(9,5))

ls = np.arange(0, lmax+1)
t  = lambda l, v : l*(l+1.)/(2.*np.pi)*v
p  = pl.plot

p( ls, t(ls, cltt), color='k', label='(WMAP 7 TT)' )
p( ls, t(ls, cltt * fsky_tt), color='k', linestyle='--', label=r'(WMAP 7 TT)$\cdot f_{\rm sky}$')

p( ls, t(ls, cltt**2 * cltt_bar_dat_qc), label='(data qc)', color='g')

p( ls, t(ls, cltt**2 * cltt_bar_sim_avg_qc.avg()), label='(sims qc)', color='r')
pl.ylabel(r"$l(l+1) [C_l^{TT}]^2 \bar{C}_l^{TT} / 2\pi$")

pl.legend(loc='lower left')
pl.setp(pl.gca().get_legend().get_frame(), visible=False)
pl.figtext( 0.5, 0.95, r"$C_l$ " + prefix, ha='center', fontsize=18)

pl.xlim(2, lmax)

pl.show()
