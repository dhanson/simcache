import os, tarfile
import numpy  as np
import healpy as hp

import qcinv

import simcache

basedir = os.environ.get('SIMCACHE_DATA', os.path.dirname(simcache.__file__)) + "/wmap/"

def y2r(year):
    return {7 : 4, 9 : 5}[year]

def n2r(nside):
    return {512 : 9}[nside]

#

wmap_das   = ['K1', 'Ka1', 'Q1', 'Q2', 'V1', 'V2', 'W1', 'W2', 'W3', 'W4']
wmap_bands = ['K', 'Ka', 'Q', 'V', 'W']

#
sigma0 = {} # (year, forered, T or P) tuples.
sigma0[(7, False, 'T')] = { 'K1' : 1.437, 'Ka1' : 1.470,
                            'Q1' : 2.254, 'Q2'  : 2.140,
                            'V1' : 3.319, 'V2'  : 2.955,
                            'W1' : 5.906, 'W2'  : 6.572, 'W3' : 6.941, 'W4' : 6.778,
                            'K'  : 1.437, 'Ka'  : 1.470, 'Q'  : 2.197, 'V'  : 3.137, 'W' : 6.549 }
sigma0[(7, True, 'T')]  = sigma0[(7, False, 'T')]

sigma0[(9, False, 'T')] = { 'K1' : 1.429, 'Ka1' : 1.466,
                            'Q1' : 2.245, 'Q2'  : 2.131,
                            'V1' : 3.314, 'V2'  : 2.949,
                            'W1' : 5.899, 'W2'  : 6.562, 'W3' : 6.941, 'W4' : 6.773,
                            'K'  : 1.429, 'Ka'  : 1.466, 'Q'  : 2.188, 'V'  : 3.131, 'W' : 6.544 }
sigma0[(9, True, 'T')]  = sigma0[(9, False, 'T')]

sigma0[(7, False, 'P')] = { 'K1' : 1.456, 'Ka1' : 1.490,
                            'Q1' : 2.290, 'Q2'  : 2.164,
                            'V1' : 3.348, 'V2'  : 2.979,
                            'W1' : 5.940, 'W2'  : 6.612, 'W3' : 6.983, 'W4' : 6.840,
                            'K'  : 1.456, 'Ka'  : 1.490, 'Q'  : 2.227, 'V' : 3.164, 'W' : 6.594 }

sigma0[(7, True, 'P')] = {  'K1' : 0.000, 'Ka1' : 2.192,
                            'Q1' : 2.741, 'Q2'  : 2.602,
                            'V1' : 3.567, 'V2'  : 3.174,
                            'W1' : 6.195, 'W2'  : 6.896, 'W3' : 7.283, 'W4' : 7.134,
                            'K'  : 0.000, 'Ka'  : 2.192, 'Q'  : 2.672, 'V' : 3.371, 'W' : 6.877 }

def get_fname_iqumap(year, det, forered):
    assert( year in [7,9] ) #FIXME: remove after sanity checking that other years will work.
    if det in ['K', 'K1']:
        assert(forered == False)

    tdname = ("/data/map/dr" + str(y2r(year)) +
              "/skymaps/" + str(year) + "yr/" +
              {False : 'raw/', True : 'forered/'}[forered == True])
    
    tfname = ("wmap" +
              {False : '', True : '_da'}[det in wmap_das] +
              {False : '', True : '_band'}[det in wmap_bands] + 
              {False : '', True : '_forered'}[forered == True] +
              "_iqumap_r9_" + str(year) + "yr_" + det + "_v" + str(y2r(year)) + ".fits")

    rfname = basedir + tdname + tfname

    if not os.path.exists(rfname):
        if not os.path.exists(basedir + tdname): os.makedirs(basedir + tdname)
        url = "http://lambda.gsfc.nasa.gov" + tdname + tfname
        simcache.util.download(url, rfname)      
    assert(os.path.exists(rfname))

    return rfname

def get_fname_imap(year, det, forered):
    assert( year in [7,9] ) #FIXME: remove after sanity checking that other years will work.
    if det in ['K', 'K1']:
        assert(forered == False)

    tdname = ("/data/map/dr" + str(y2r(year)) +
              "/skymaps/" + str(year) + "yr/" +
              {False : 'raw/', True : 'forered/'}[forered == True])
    
    tfname = ("wmap" +
              {False : '', True : '_da'}[det in wmap_das] +
              {False : '', True : '_band'}[det in wmap_bands] + 
              {False : '', True : '_forered'}[forered == True] +
              "_imap_r9_" + str(year) + "yr_" + det + "_v" + str(y2r(year)) + ".fits")

    rfname = basedir + tdname + tfname

    if not os.path.exists(rfname):
        if not os.path.exists(basedir + tdname): os.makedirs(basedir + tdname)
        url = "http://lambda.gsfc.nasa.gov" + tdname + tfname
        simcache.util.download(url, rfname)      
    assert(os.path.exists(rfname))

    return rfname
    
def get_fname_temperature_analysis_mask(year, nside, label=None):
    assert(year in [7,9]) #FIXME: remove after sanity checking that other years will work.
    assert(nside == 512)  #FIXME: remove after sanity checking that other nside will work. 

    if label == None:
        label = ''
    else:
        label = label + '_'
    
    tdname = ("/data/map/dr" + str(y2r(year)) +
              "/ancillary/masks/")
    tfname = ("wmap_temperature_" + label + "analysis_mask_r" + str(n2r(nside)) +  "_" + str(year) + "yr_v" + str(y2r(year)) + ".fits")

    rfname = basedir + tdname + tfname

    if not os.path.exists(rfname):
        if not os.path.exists(basedir + tdname): os.makedirs(basedir + tdname)
        url = "http://lambda.gsfc.nasa.gov" + tdname + tfname
        simcache.util.download(url, rfname)
    assert(os.path.exists(rfname))

    return rfname

def get_fname_polarization_analysis_mask(year, nside):
    assert(year in [7,9]) #FIXME: remove after sanity checking that other years will work.
    assert(nside == 512)  #FIXME: remove after sanity checking that other nside will work. 
    
    tdname = ("/data/map/dr" + str(y2r(year)) +
              "/ancillary/masks/")
    tfname = ("wmap_polarization_analysis_mask_r" + str(n2r(nside)) +  "_" + str(year) + "yr_v" + str(y2r(year)) + ".fits")

    rfname = basedir + tdname + tfname

    if not os.path.exists(rfname):
        if not os.path.exists(basedir + tdname): os.makedirs(basedir + tdname)
        url = "http://lambda.gsfc.nasa.gov" + tdname + tfname
        simcache.util.download(url, rfname)
    assert(os.path.exists(rfname))

    return rfname

def get_fname_bestfit_cl(year, label):
    assert(year in [7,9]) #FIXME: remove after sanity checking that other years will work.

    # http://lambda.gsfc.nasa.gov/data/map/dr4/dcp/c_ls/wmap_lcdm_sz_lens_wmap7_cl_v4.tar.gz
    
    tdname = ("/data/map/dr" + str(y2r(year)) +
              "/dcp/c_ls/")
    tfname = ("wmap_" + label +  "_wmap" + str(year) + "_cl_v" + str(y2r(year)))

    rfname = basedir + tdname + tfname

    if not os.path.exists(rfname):
        if not os.path.exists(basedir + tdname): os.makedirs(basedir + tdname)
        url = "http://lambda.gsfc.nasa.gov" + tdname + tfname + ".tar.gz"
        simcache.util.download(url, rfname + ".tar.gz")
        tf = tarfile.open(rfname + ".tar.gz", "r:gz")
        tf.extractall(rfname)
    
    assert(os.path.exists(rfname))

    return rfname

def get_bl(year, det):
    assert(year in [7,9])    #FIXME: remove after sanity checking that other years will work.
    
    if det in wmap_das:
        tdname = ("/data/map/dr" + str(y2r(year)) +
                  "/ancillary/beams/")
        tfname = ("wmap_ampl_bl_" + det + "_" + str(year) + "yr_v" + str(y2r(year)) + ".txt")

        rfname = basedir + tdname + tfname
        
        if not os.path.exists(rfname):
            if not os.path.exists(basedir + tdname): os.makedirs(basedir + tdname)
            url = "http://lambda.gsfc.nasa.gov" + tdname + tfname
            simcache.util.download(url, rfname)
        assert(os.path.exists(rfname))

        return np.loadtxt(rfname)[:,1]
    
    elif det in wmap_bands:
        if (det == 'K'):
            return get_bl(year, 'K1')
        elif (det == 'Ka'):
            return get_bl(year, 'Ka1')
        if (det == 'Q'):
            return 0.5 * (get_bl(year, 'Q1') + get_bl(year, 'Q2'))
        elif (det == 'V'):
            return 0.5 * (get_bl(year, 'V1') + get_bl(year, 'V2'))
        elif (det == 'W'):
            return 0.25 * (get_bl(year, 'W1') + get_bl(year, 'W2') + get_bl(year, 'W3') + get_bl(year, 'W4'))
    else:
        assert(0)

def get_bestfit_cl_scalar(year, label, lmax=None):
    rfname = get_fname_bestfit_cl(year, label)
    return qcinv.util.camb_clfile(rfname + "/bestfit_scalCls.dat", lmax=lmax)

def get_bestfit_cl_lensed(year, label, lmax=None):
    rfname = get_fname_bestfit_cl(year, label)
    return qcinv.util.camb_clfile(rfname + "/bestfit_lensedCls.dat", lmax=lmax)

def get_nlev_t_uK_arcmin(year, det, forered, mask_t):
    nobsinv = 1.0 / hp.read_map( get_fname_imap(year, det, forered), hdu=1, field=1 )

    npix = len(nobsinv)
    nlev_t = np.sqrt( 4.*np.pi/npix * np.sum( mask_t * nobsinv * sigma0[(year, forered, 'T')][det]**2 * 1.e6  ) / np.sum(mask_t) ) * 180.*60./np.pi

    return nlev_t

def get_nlev_tp_uK_arcmin(year, det, forered, mask_t, mask_p):
    nobsinv = 1.0 / hp.read_map( get_fname_iqumap(year, det, forered), hdu=1, field=3 )

    npix = len(nobsinv)
    nlev_t = np.sqrt( 4.*np.pi/npix * np.sum( mask_t * nobsinv * sigma0[(year, forered, 'T')][det]**2 * 1.e6  ) / np.sum(mask_t) ) * 180.*60./np.pi
    nlev_p = np.sqrt( 4.*np.pi/npix * np.sum( mask_p * nobsinv * sigma0[(year, forered, 'P')][det]**2 * 1.e6  ) / np.sum(mask_p) ) * 180.*60./np.pi * np.sqrt(2.)

    return nlev_t, nlev_p

def get_cl(year, id):
    assert(year in [7,9])

    
