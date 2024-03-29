#!/usr/bin/env python

import os, imp, sys, time, getopt
import numpy as np
import pylab as pl

import qcinv
import simcache

import simcache.sims_teblm        as sims_teblm
import simcache.wmap.sims_tpmap   as sims_tpmap
import simcache.wmap.filt_ivf_teb as filt_ivf_teb

prefix  = "sims_tp"

det     = 'V'
year    = 7
forered = True
ntype   = 'nobs_qucov'
lmax    = 1000
nside   = 512

nsims   = 10

mask_t  = simcache.wmap.dmc.get_fname_temperature_analysis_mask(year, nside)
mask_p  = simcache.wmap.dmc.get_fname_polarization_analysis_mask(year, nside)

cl = simcache.wmap.dmc.get_bestfit_cl_lensed(7, 'lcdm_sz_lens')
bl = simcache.wmap.dmc.get_bl(7, 'V')[0:lmax+1]

# sim params.
sims_cmb = sims_teblm.cmb_library(lmax, cl, "scratch/teblm_cmb_library_wmap")
sims     = sims_tpmap.cmb_nse_library(lmax, nside, year, sims_cmb, forered, ntype=ntype, lib_dir=("scratch/" + prefix + "/sims"))

# ivf params.
ivfs_qc    = filt_ivf_teb.qcinv_library(lmax, cl, mask_t, mask_p, sims, lib_dir=("scratch/" + prefix + "/ivfs_qc"))
ivfs_fl    = filt_ivf_teb.fl_library(lmax, cl, mask_t, mask_p, sims, lib_dir=("scratch/" + prefix + "/ivfs_fl"))

# -- spectra
if not os.path.exists("scratch/" + prefix):
    os.makedirs("scratch/" + prefix)

cltt = cl.cltt[0:lmax+1]
clee = cl.clee[0:lmax+1]
clbb = cl.clbb[0:lmax+1]
clte = cl.clte[0:lmax+1]

# -- mask stats
assert( ivfs_qc.get_fsky() == ivfs_fl.get_fsky() )
fsky_tt, fsky_tp, fsky_pp = ivfs_qc.get_fsky()
# --

t0 = time.time()

# -- load data
cltt_bar_dat_fl = qcinv.util_alm.alm_cl( ivfs_fl.get_dat_tlm(det) ); cltt_bar_dat_qc = qcinv.util_alm.alm_cl( ivfs_qc.get_dat_tlm(det) )
clee_bar_dat_fl = qcinv.util_alm.alm_cl( ivfs_fl.get_dat_elm(det) ); clee_bar_dat_qc = qcinv.util_alm.alm_cl( ivfs_qc.get_dat_elm(det) )
clbb_bar_dat_fl = qcinv.util_alm.alm_cl( ivfs_fl.get_dat_blm(det) ); clbb_bar_dat_qc = qcinv.util_alm.alm_cl( ivfs_qc.get_dat_blm(det) )
clte_bar_dat_fl = qcinv.util_alm.alm_cl_cross( ivfs_fl.get_dat_tlm(det), ivfs_fl.get_dat_elm(det) ); clte_bar_dat_qc = qcinv.util_alm.alm_cl_cross( ivfs_qc.get_dat_tlm(det), ivfs_qc.get_dat_elm(det) )

# -- load sims
cltt_bar_sim_avg_fl = simcache.util.avg(); cltt_bar_sim_avg_qc = simcache.util.avg()
clee_bar_sim_avg_fl = simcache.util.avg(); clee_bar_sim_avg_qc = simcache.util.avg()
clbb_bar_sim_avg_fl = simcache.util.avg(); clbb_bar_sim_avg_qc = simcache.util.avg()
clte_bar_sim_avg_fl = simcache.util.avg(); clte_bar_sim_avg_qc = simcache.util.avg()
for i in xrange(0, nsims):
    print 'loading sim i = ', i, ' elapsed =  %0.3f s' % (time.time()-t0)
    cltt_bar_sim_avg_fl += qcinv.util_alm.alm_cl( ivfs_fl.get_sim_tlm(det, i) )
    clee_bar_sim_avg_fl += qcinv.util_alm.alm_cl( ivfs_fl.get_sim_elm(det, i) )
    clbb_bar_sim_avg_fl += qcinv.util_alm.alm_cl( ivfs_fl.get_sim_blm(det, i) )
    clte_bar_sim_avg_fl += qcinv.util_alm.alm_cl_cross( ivfs_fl.get_sim_tlm(det, i), ivfs_fl.get_sim_elm(det, i) )

    cltt_bar_sim_avg_qc += qcinv.util_alm.alm_cl( ivfs_qc.get_sim_tlm(det, i) )
    clee_bar_sim_avg_qc += qcinv.util_alm.alm_cl( ivfs_qc.get_sim_elm(det, i) )
    clbb_bar_sim_avg_qc += qcinv.util_alm.alm_cl( ivfs_qc.get_sim_blm(det, i) )
    clte_bar_sim_avg_qc += qcinv.util_alm.alm_cl_cross( ivfs_qc.get_sim_tlm(det, i), ivfs_qc.get_sim_elm(det, i) )

# -- make plots
pl.figure(figsize=(18,10))

ls = np.arange(0, lmax+1)
t  = lambda l, v : l*(l+1.)/(2.*np.pi)*v
p  = pl.plot

pl.subplots_adjust(left=0.05, right=0.975, bottom=0.05)

pl.subplot(221)
p( ls, t(ls, cltt), color='k', label='(WMAP 7 TT)' )
p( ls, t(ls, cltt * fsky_tt), color='k', linestyle='--', label=r'(WMAP 7 TT)$\cdot f_{\rm sky}$')

p( ls, t(ls, cltt**2 * cltt_bar_dat_fl), label='(data fl)', color='m')
p( ls, t(ls, cltt**2 * cltt_bar_dat_qc), label='(data qc)', color='g')

p( ls, t(ls, cltt**2 * cltt_bar_sim_avg_fl.avg()), label='(sims fl)', color='b')
p( ls, t(ls, cltt**2 * cltt_bar_sim_avg_qc.avg()), label='(sims qc)', color='r')
pl.ylabel(r"$l(l+1) [C_l^{TT}]^2 \bar{C}_l^{TT} / 2\pi$")

pl.subplot(222)

p( ls, t(ls, clte**2 * clte_bar_dat_fl), label='(data fl)', color='m')
p( ls, t(ls, clte**2 * clte_bar_dat_qc), label='(data qc)', color='g')

p( ls, t(ls, clte**2 * clte_bar_sim_avg_fl.avg()), label='(sims fl)', color='b')
p( ls, t(ls, clte**2 * clte_bar_sim_avg_qc.avg()), label='(sims qc)', color='r')
pl.ylabel(r"$l(l+1) \bar{C}_l^{TE} / 2\pi$")

pl.subplot(223)
p( ls, t(ls, clee**2 * clee_bar_dat_fl), label='(data fl)', color='m')
p( ls, t(ls, clee**2 * clee_bar_dat_qc), label='(data qc)', color='g')

p( ls, t(ls, clee**2 * clee_bar_sim_avg_fl.avg()), label='(sims fl)', color='b')
p( ls, t(ls, clee**2 * clee_bar_sim_avg_qc.avg()), label='(sims qc)', color='r')
pl.ylabel(r"$l(l+1) \bar{C}_l^{EE} / 2\pi$")

pl.subplot(224)
p( ls, t(ls, clbb_bar_dat_fl), label='(data fl)', color='m')
p( ls, t(ls, clbb_bar_dat_qc), label='(data qc)', color='g')

p( ls, t(ls, clbb_bar_sim_avg_fl.avg()), label='(sims fl)', color='b')
p( ls, t(ls, clbb_bar_sim_avg_qc.avg()), label='(sims qc)', color='r')
pl.ylabel(r"$l(l+1) \bar{C}_l^{BB} / 2\pi$")

pl.legend(loc='lower left')
pl.setp(pl.gca().get_legend().get_frame(), visible=False)
pl.figtext( 0.5, 0.95, r"$C_l$ " + prefix, ha='center', fontsize=18)

pl.xlim(2, lmax)

pl.show()
