import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import os
import sys
import logging
from astropy.io import fits
from multiprocessing import Pool
import pandas as pd

params = {'font.size'     : 14,
          'figure.figsize':(15.0, 8.0),
          'lines.linewidth': 2.,
          'lines.markersize': 15,}
mpl.rcParams.keys()
mpl.rcParams.update(params)
np.set_printoptions(suppress=True)

############### iSpec import stuff ######################

# specify ispec dir so iSpec can run binaries for code synthesis
ispec_dir = '/home/lazar/Fak(s)/prakse/WRO/iSpec_v20180608'
sys.path.insert(0,os.path.abspath(ispec_dir)) #append to path

import ispec

LOG_LEVEL = "warning" #you can set it to info, but its noisy
logger = logging.getLogger()
logger.setLevel(logging.getLevelName(LOG_LEVEL.upper()))
################################################################################

#Specify where is your data and read tsv containing object names
data_root = '/home/lazar/Fak(s)/prakse/WRO/random_data/data_m67/'
pandas_table = pd.read_table(data_root+'m67_cat.tsv')
#select only object_id
selected_star_cname = pandas_table["CNAME"][0:2]
#specify fits root
fits_root = data_root+'M67_NGC2682/'


with open(fits_root+'izlaz') as f:
    con = f.readlines()

#get rid of "\n" and append fits_root
fits_names = [x.strip() for x in con]
fits_path = [fits_root + s for s in fits_names]
#print(fits_path[0])


#get all obj_id from fits header from root files
obj_id = list(map(lambda fp: fits.open(fp)[0].header['object'], fits_path))
#simple check to see if everything is ok
#print(obj_id[0])

################################################################################
####################### HELPER FUNCTIONS########################################
################################################################################

def argsort(seq):
    ''' for some sequence, returne arguments such that list is sorted '''
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__)

def get_files_for_obj_id(obj_id, obj_list, file_list):
    ''' input one object from obj_list and returns coresponding files
    for that object'''
    indexes = [i for i,x in enumerate(obj_list) if x == obj_id]
    return [file_list[x] for x in indexes]

def read_wave_and_return_sorted(file_list):
    '''For multiple files reads both fits, compare min wave and returns
    filepath for lower star first'''
    wave_ln = list(map(lambda x: fits.open(x)[0].header['WAVELMIN'], file_list))
    return [file_list[x] for x in argsort(wave_ln)]

def get_merged_spectra(sorted_file_list):
    """if you have multiple files for one object
    this procedure merges it and returns one spectra
    using ispec tools"""
    spectras = map(lambda x: ispec.read_spectrum(x), sorted_file_list)
    return np.hstack(([x for x in spectras]))


################################################################################
####################### iSpec wrap func ########################################
################################################################################

def estimate_snr_from_err(star_spectrum):
    ''' Estimate snr from error column in star_spectrum iSpec object '''
    logging.info("Estimating SNR from errors...")
    efilter = star_spectrum['err'] > 0
    filtered_star_spectrum = star_spectrum[efilter]
    if len(filtered_star_spectrum) > 1:
        estimated_snr = np.median(filtered_star_spectrum['flux'] / filtered_star_spectrum['err'])
    else:
        print('All the errors are set to zero and we cannot calculate SNR using them\n')
        estimated_snr = 0
    return estimated_snr

########## pre defined function, IDK why used wraper here #####################
def estimate_snr_from_flux(star_spectrum, num_points=10):
    ''' Estimate snr from flux ignoring errors '''
    logging.info("Estimating SNR from fluxes...")
    snr = ispec.estimate_snr(star_spectrum['flux'], num_points=num_points)
    return snr


def clean_telluric_regions(star_spectrum, replace_mask_zero = False):
    ''' Clean telluric regions from given spectrum and return clean spectrum and
    velocity offset '''
    logging.info("Telluric velocity shift determination...")
    # - Read telluric linelist
    telluric_linelist_file = ispec_dir + \
            "/input/linelists/CCF/Synth.Tellurics.500_1100nm/mask.lst"
    telluric_linelist = ispec.read_telluric_linelist(\
            telluric_linelist_file, minimum_depth=0.0)


    # find cross correlation function of star_spectrum and selected mask
    # in this case, mask is telluric_line_mask
    # with preset limit velocity
    # and find extreme values
    models, ccf = ispec.cross_correlate_with_mask(star_spectrum, \
            telluric_linelist, lower_velocity_limit=-100, \
            upper_velocity_limit=100, velocity_step=0.5, \
            mask_depth=0.01, fourier = False, only_one_peak = True)

    #round down to 2 decimals
    bv = np.round(models[0].mu(), 2) # km/s
    bv_err = np.round(models[0].emu(), 2) # km/s


    # - Filter regions that may be affected by telluric lines
    #bv = 0.0
    min_vel = -30.0
    max_vel = +30.0
    # Only the 25% of the deepest ones:
    dfilter = telluric_linelist['depth'] > np.percentile(telluric_linelist['depth'], 75)
    tfilter = ispec.create_filter_for_regions_affected_by_tellurics(\
            star_spectrum['waveobs'], telluric_linelist[dfilter], \
            min_velocity=-bv+min_vel, max_velocity=-bv+max_vel)
    clean_star_spectrum = star_spectrum[~tfilter]
    
    #if you want to replace removed flux with 0
    if replace_mask_zero:
        clean_star_spectrum = star_spectrum.copy() 
        clean_star_spectrum['flux'][tfilter] = 0

    return clean_star_spectrum, bv

def determine_radial_velocity_with_template(star_spectrum):
    ''' Determine radial velocity of star_spectrum compared with some template
    Chose template wisely; Return Radial velocity and err'''
    logging.info("Radial velocity determination with template...")
    # - Read synthetic template
    #template = ispec.read_spectrum(ispec_dir + "/input/spectra/templates/Atlas.Arcturus.372_926nm/template.txt.gz")
    #template = ispec.read_spectrum(ispec_dir + "/input/spectra/templates/Atlas.Sun.372_926nm/template.txt.gz")
    template = ispec.read_spectrum(ispec_dir + "/input/spectra/templates/NARVAL.Sun.370_1048nm/template.txt.gz")
    #template = ispec.read_spectrum(ispec_dir + "/input/spectra/templates/Synth.Sun.300_1100nm/template.txt.gz")

    models, ccf = ispec.cross_correlate_with_template(star_spectrum, template,\
            lower_velocity_limit=-200, upper_velocity_limit=200,\
            velocity_step=1.0, fourier=False)

    # First component:
    rv = np.round(models[0].mu(), 2) # km/s
    rv_err = np.round(models[0].emu(), 2) # km/s
    return rv, rv_err

##### NO NEED FOR THIS WRAPER, CHANGE FUNCTION CALL BELLOW

def correct_radial_velocity(star_spectrum, rv):
    ''' correct radial velocity'''
    logging.info("Radial velocity correction...")
    #rv = -96.40 # km/s
    return ispec.correct_velocity(star_spectrum, rv)


def normalize_spectrum(star_spectrum, res=80000, model = "Splines", degree=2, nknots=None):
    '''
    Fit continuum ignoring strong lines and using fe lines template, 
    strategy 'median+max'
    ####
    #### ASSUMING DEFAULT VALUES FOR:
    #### RESOLUTION, MODEL FOR CONTINUUM, DEGREE OF MODEL, NKNOTS!!!!
    #### TAKE CARE!!!!
    ####
    returns normalized star spectrum object and continumm model
    '''
    from_resolution = res #define resolution

    # Strategy: Filter first median values and secondly MAXIMUMs in order to find the continuum
    order='median+max'
    median_wave_range=0.05
    max_wave_range=1.0
    # read continuum regions from Fe lines and strong lines

    continuum_regions = ispec.read_continuum_regions(\
            ispec_dir + "/input/regions/fe_lines_continuum.txt")
    strong_lines = ispec.read_line_regions(\
            ispec_dir + "/input/regions/strong_lines/absorption_lines.txt")
    
    star_continuum_model = ispec.fit_continuum(star_spectrum,\
            from_resolution=from_resolution, ignore=strong_lines,\
            continuum_regions=continuum_regions, nknots=nknots, degree=degree,\
            median_wave_range=median_wave_range, max_wave_range=max_wave_range,\
            model=model, order=order, automatic_strong_line_detection=True,\
            strong_line_probability=0.5, use_errors_for_fitting=True)

    #--- Continuum normalization -----------------------------------------------
    
    logging.info("Continuum normalization...")
    normalized_star_spectrum = ispec.normalize_spectrum(star_spectrum,\
            star_continuum_model, consider_continuum_errors=False)
    
    #Use a fixed value because the spectrum is already normalized
    #you need continuum model to be able to synt or interpolate params from grid
    star_continuum_model = ispec.fit_continuum(star_spectrum, \
            fixed_value=1.0, model="Fixed value")
    return normalized_star_spectrum, star_continuum_model

def param_using_grid(normalized_star_spectrum, star_continuum_model, object_id,\
    resolution=80000, p0 = [5750.0, 4.5, 0, 0, 2, 0.6, 0], max_iter = 10):
    ''' Derive spectroscopic parameters using grid model 
    p0 are initial values list:
    teff, logg, MH, alpha, vsini, limb_darkening coef, vrad 
   
    Returns params, errors and ispec_pectrum object with spectra form best fit
    '''

    #--- Model spectra --------------------------------------------------------
    # Parameters initiall values
    initial_teff = p0[0]
    initial_logg = p0[1]
    initial_MH = p0[2]
    initial_alpha = p0[3]
    initial_vmic = ispec.estimate_vmic(initial_teff, initial_logg, initial_MH)
    initial_vmac = ispec.estimate_vmac(initial_teff, initial_logg, initial_MH)
    initial_vsini = p0[4]
    initial_limb_darkening_coeff = p0[5]
    initial_R = resolution
    initial_vrad = p0[6]
    max_iterations = max_iter

    #load grid
    code = "grid"
    precomputed_grid_dir = ispec_dir + "/input/grid/SPECTRUM_MARCS.GES_GESv5_atom_hfs_iso.480_680nm_light/"

    atomic_linelist = None
    isotopes = None
    modeled_layers_pack = None
    solar_abundances = None
    free_abundances = None
    linelist_free_loggf = None

    # Free parameters (vmic cannot be used as a free parameter when using a spectral grid)
    
    free_params = ["teff", "logg", "MH", "alpha", "R"]

    # Line regions
    line_regions = ispec.read_line_regions(ispec_dir + "/input/regions/47000_GES/{}_synth_good_for_params_all.txt".format(code))
    #line_regions = ispec.read_line_regions(ispec_dir + "/input/regions/47000_GES/{}_synth_good_for_params_all_extended.txt".format(code))
    #line_regions = ispec.read_line_regions(ispec_dir + "/input/regions/47000_VALD/{}_synth_good_for_params_all.txt".format(code))
    #line_regions = ispec.read_line_regions(ispec_dir + "/input/regions/47000_VALD/{}_synth_good_for_params_all_extended.txt".format(code))
    ## Select only some lines to speed up the execution (in a real analysis it is better not to do this)
    #line_regions = line_regions[np.logical_or(line_regions['note'] == 'Ti 1', line_regions['note'] == 'Ti 2')]
    #line_regions = ispec.adjust_linemasks(normalized_star_spectrum, line_regions, max_margin=0.5)
    
    # Read segments if we have them or...
    #segments = ispec.read_segment_regions(ispec_dir + "/input/regions/fe_lines_segments.txt")
    # ... or we can create the segments on the fly:
    segments = ispec.create_segments_around_lines(line_regions, margin=0.25)

    ## Add also regions from the wings of strong lines:
    # H beta
    hbeta_lines = ispec.read_line_regions(ispec_dir + "/input/regions/wings_Hbeta.txt")
    hbeta_segments = ispec.read_segment_regions(ispec_dir + "/input/regions/wings_Hbeta_segments.txt")
    line_regions = np.hstack((line_regions, hbeta_lines))
    segments = np.hstack((segments, hbeta_segments))
    # H alpha
    halpha_lines = ispec.read_line_regions(ispec_dir + "/input/regions/wings_Halpha.txt")
    halpha_segments = ispec.read_segment_regions(ispec_dir + "/input/regions/wings_Halpha_segments.txt")
    line_regions = np.hstack((line_regions, halpha_lines))
    segments = np.hstack((segments, halpha_segments))
    # Magnesium triplet
    mgtriplet_lines = ispec.read_line_regions(ispec_dir + "/input/regions/wings_MgTriplet.txt")
    mgtriplet_segments = ispec.read_segment_regions(ispec_dir + "/input/regions/wings_MgTriplet_segments.txt")
    line_regions = np.hstack((line_regions, mgtriplet_lines))
    segments = np.hstack((segments, mgtriplet_segments))


    # run model spectra from grid!
    obs_spec, modeled_synth_spectrum, params, errors, abundances_found,\
    loggf_found, status, stats_linemasks = \
    ispec.model_spectrum(normalized_star_spectrum, star_continuum_model,\
    modeled_layers_pack, atomic_linelist, isotopes, solar_abundances,\
    free_abundances, linelist_free_loggf, initial_teff, initial_logg,\
    initial_MH, initial_alpha, initial_vmic, initial_vmac, initial_vsini,\
    initial_limb_darkening_coeff, initial_R, initial_vrad, free_params,\
    segments=segments, linemasks=line_regions, enhance_abundances=False,\
    use_errors = True, vmic_from_empirical_relation = False,\
    vmac_from_empirical_relation = True, max_iterations=max_iterations,\
    tmp_dir = None, code=code, precomputed_grid_dir=precomputed_grid_dir)

    ##--- Save results ---------------------------------------------------------
    #logging.info("Saving results...")
    dump_file = "example_results_synth_grid_%s.dump" % (object_id)
    #logging.info("Saving results...")
    ispec.save_results(dump_file, (params, errors, abundances_found, loggf_found, status, stats_linemasks))
    # If we need to restore the results from another script:
    # params, errors, abundances_found, loggf_found, status, stats_linemasks = ispec.restore_results(dump_file)

    #logging.info("Saving synthetic spectrum...")
    synth_filename = "example_modeled_synth_grid_%s.fits" % (object_id)
    ispec.write_spectrum(modeled_synth_spectrum, synth_filename)
    return params, errors, modeled_synth_spectrum


def abund_line_by_line(star_spectrum, param, star_continuum_model, object_id,code="grid"):
    ''' more or less same function as one above but i created wraper
    just to make things clear;
    The idea is that we use grid interpolation but only free param is metalicity
    and to fix everything else from previously derived model atmh;
    Drawback is it only returns metalicity, not abundance of elements
    because you cant interpolate abundances in grid
    But with line mask and fitting by segment, metalicity you get is bassicaly
    abundance for that spectral line; One elements creates multiple line
    so just take averages afterwards and you can use that for rough estimate'''

    normalized_star_spectrum = star_spectrum
    precomputed_grid_dir = ispec_dir + "/input/grid/SPECTRUM_MARCS.GES_GESv5_atom_hfs_iso.480_680nm_light/"

    #--- Model spectra ----------------------------------------------------------
    # Parameters
    initial_teff = param['teff']
    initial_logg = param['logg']
    initial_MH = param['MH']
    initial_alpha = param['alpha']
    initial_vmic = param['vmic']
    initial_vmac = param['vmac']
    initial_vsini = param['vsini']
    initial_limb_darkening_coeff = param['limb_darkening_coeff']
    initial_R = param['R']
    initial_vrad = 0
    max_iterations = 10

    # Selected model amtosphere, linelist and solar abundances
    #model = ispec_dir + "/input/atmospheres/MARCS/"
    model = ispec_dir + "/input/atmospheres/MARCS.GES/"
    #model = ispec_dir + "/input/atmospheres/MARCS.APOGEE/"
    #model = ispec_dir + "/input/atmospheres/ATLAS9.APOGEE/"
    #model = ispec_dir + "/input/atmospheres/ATLAS9.Castelli/"
    #model = ispec_dir + "/input/atmospheres/ATLAS9.Kurucz/"
    #model = ispec_dir + "/input/atmospheres/ATLAS9.Kirby/"

    #atomic_linelist_file = ispec_dir + "/input/linelists/transitions/VALD.300_1100nm/atomic_lines.tsv"
    #atomic_linelist_file = ispec_dir + "/input/linelists/transitions/VALD.1100_2400nm/atomic_lines.tsv"
    atomic_linelist_file = ispec_dir + "/input/linelists/transitions/GESv5_atom_hfs_iso.420_920nm/atomic_lines.tsv"
    #atomic_linelist_file = ispec_dir + "/input/linelists/transitions/GESv5_atom_nohfs_noiso.420_920nm/atomic_lines.tsv"

    solar_abundances_file = ispec_dir + "/input/abundances/Grevesse.2007/stdatom.dat"
    #solar_abundances_file = ispec_dir + "/input/abundances/Asplund.2005/stdatom.dat"
    #solar_abundances_file = ispec_dir + "/input/abundances/Asplund.2009/stdatom.dat"
    #solar_abundances_file = ispec_dir + "/input/abundances/Grevesse.1998/stdatom.dat"
    #solar_abundances_file = ispec_dir + "/input/abundances/Anders.1989/stdatom.dat"

    isotope_file = ispec_dir + "/input/isotopes/SPECTRUM.lst"

    # Load chemical information and linelist
    atomic_linelist = ispec.read_atomic_linelist(\
            atomic_linelist_file, wave_base=np.min(star_spectrum['waveobs']),\
            wave_top=np.max(star_spectrum['waveobs']))
    # Select lines that have some minimal contribution in the sun
    atomic_linelist = atomic_linelist[atomic_linelist['theoretical_depth'] >= 0.01] 

    isotopes = ispec.read_isotope_data(isotope_file)



    # Load model atmospheres
    modeled_layers_pack = ispec.load_modeled_layers_pack(model)

    # Load SPECTRUM abundances
    solar_abundances = ispec.read_solar_abundances(solar_abundances_file)


    # Free parameters
    #free_params = ["teff", "logg", "MH", "vmic", "vmac", "vsini", "R", "vrad", "limb_darkening_coeff"]
    #free_params = ["vrad"]
    free_params = ["MH"]
    #this is where we fix drawback; if we use synth we could use abundances
    free_abundances = None
    # Free individual element abundance (WARNING: it should be coherent with the selected line regions!)

    chemical_elements_file = ispec_dir + "/input/abundances/chemical_elements_symbols.dat"   
    chemical_elements = ispec.read_chemical_elements(chemical_elements_file)

    # Line regions
    line_regions = ispec.read_line_regions(ispec_dir + "/input/regions/47000_GES/grid_synth_good_for_params_all.txt".format(code))
    #line_regions = ispec.read_line_regions(ispec_dir + "/input/regions/47000_GES/{}_synth_good_for_params_all_extended.txt".format(code))
    #line_regions = ispec.read_line_regions(ispec_dir + "/input/regions/47000_VALD/{}_synth_good_for_params_all.txt".format(code))
    #line_regions = ispec.read_line_regions(ispec_dir + "/input/regions/47000_VALD/{}_synth_good_for_params_all_extended.txt".format(code))
    
    # Select only the lines to get abundances from
    #line_regions = line_regions[0:5]
    line_regions = ispec.adjust_linemasks(normalized_star_spectrum, line_regions, max_margin=0.5)

    output_dirname = "example_abundance_line_by_line_%s" % (code,)
    ispec.mkdir_p(output_dirname)
    abund_array = np.array(())
    #create empty text file


    for i, line in enumerate(line_regions):
        # Directory and file names
        #element_name = "_".join(line['element'].split())
        element_name = "_".join(line['note'].split())
        common_filename = "example_" + code + "_individual_" + element_name + "_%.4f" % line['wave_peak']
        print("=========ELEMENT NAME==============")
        print(element_name)
        
        linelist_free_loggf = None

        # Line by line
        individual_line_regions = line_regions[i:i+1] # Keep recarray structure

        # Segment
        segments = ispec.create_segments_around_lines(individual_line_regions, margin=0.25)
        wfilter = ispec.create_wavelength_filter(normalized_star_spectrum, regions=segments) # Only use the segment
        
        #skip this line if flux is 0 somewhere in region or there is no data

        if len(normalized_star_spectrum[wfilter]) == 0 or np.any(normalized_star_spectrum[wfilter] == 0):
            continue 

        #this sometimes fails for different reasons
        #if it does, lets ignore this line
        try:
            obs_spec, modeled_synth_spectrum, derived_params, errors,\
        abundances_found, loggf_found, status, stats_linemasks = \
        ispec.model_spectrum(normalized_star_spectrum[wfilter],\
        star_continuum_model, modeled_layers_pack, atomic_linelist,\
        isotopes, solar_abundances, free_abundances, linelist_free_loggf,\
        initial_teff, initial_logg, initial_MH, initial_alpha, initial_vmic,\
        initial_vmac, initial_vsini, initial_limb_darkening_coeff, initial_R,\
        initial_vrad, free_params, segments=segments,\
        linemasks=individual_line_regions,enhance_abundances=True,\
        use_errors = True, vmic_from_empirical_relation = False,\
        vmac_from_empirical_relation = False, max_iterations=max_iterations,\
        tmp_dir = None, code=code, precomputed_grid_dir=precomputed_grid_dir)
        
        except Exception:
            continue


        #Write every element abundance to separate file
        #We use tihs ugly stuff here because if model_spectrum fails
        #it raises exception and i dont currently know how to handle it
        abundances_file = open(output_dirname + "/abd/%s_%s_abundances.txt" %(object_id, element_name), "a")
        abundances_file.write("%f\t%f\n" %(derived_params['MH'], errors['MH']))
        abundances_file.close()
        
        
        ##--- Save results -------------------------------------------------------------
        dump_file = output_dirname + "/" + 'dumps'+ '/' + common_filename + ".dump"
        logging.info("Saving results...")
        ispec.save_results(dump_file, (derived_params, errors, abundances_found, loggf_found, status, stats_linemasks))
        # If we need to restore the results from another script:
        # params, errors, abundances_found, loggf_found, status, stats_linemasks = ispec.restore_results(dump_file)
        logging.info("Saving synthetic spectrum...")
        synth_filename = output_dirname + "/" + common_filename + ".fits"
        ispec.write_spectrum(modeled_synth_spectrum, synth_filename)
    
    return abund_array


# In[43]:


def do_stuff(what_object='08505182+1156559'):
    degrade_res=47000
    files_for_obj = get_files_for_obj_id(what_object, obj_id, fits_path)
    files_sorted = read_wave_and_return_sorted(files_for_obj)
    my_random_star = get_merged_spectra(files_sorted) #star i selected
    plt.plot(my_random_star['waveobs'],my_random_star['flux'])
    my_random_star, bv = clean_telluric_regions(my_random_star) #my clean star
    print('---------- Rv from teluric -------------')
    print(bv)
    r_v, _ = determine_radial_velocity_with_template(my_random_star)
    print('---------- Rv from template -------------')
    print(r_v)
    my_random_star = correct_radial_velocity(my_random_star, r_v)
    plt.plot(my_random_star['waveobs'],my_random_star['flux'])
    print('---------- SNR from flux -------------')
    print(estimate_snr_from_flux(my_random_star))
    print('---------- SNR from err -------------')
    print(estimate_snr_from_err(my_random_star))
    print('---------- Initial spectra resolution ---------')
    myres = fits.open(files_for_obj[0])[0].header['SPEC_RES']
    print(myres)
    print('---------- Fit continuum and normalize ---------')
    my_random_star, c_model = normalize_spectrum(my_random_star, myres)
    plt.plot(my_random_star['waveobs'],my_random_star['flux'])
    ispec.write_spectrum(my_random_star, 'normalized_spectra.fits')
    print('---------- Determine parameters using grid ---------')
    param, err, fit = param_using_grid(my_random_star, c_model, what_object, myres)
    
    #np.savetxt('%s_params' %(what_object), np.array([param,err]))
    # Crazy shit, np.save is not working
    f = open('%s_params' %(what_object), 'w')
    f.write(str(param['teff']) + '\t' + str(param['logg']) + '\t'\
            +str(param['MH']) + '\t' + str(param['alpha']) + '\t'\
            +str(param['vmic']) + '\t' + str(param['vmac']) + '\t'\
            +str(param['vsini']) + '\n' + str(err['teff']) + '\t'\
            +str(param['logg']) + '\t' + str(err['MH']) + '\t'\
            +str(param['alpha']) + '\t' + str(err['vmic']) + '\t'\
            +str(param['vmac']) + '\t' + str(err['vsini']) + '\n')
    f.close()
    
    print(param)
    print(err)
    
    #plt.close()
    #plt.plot(my_random_star['waveobs'], my_random_star['flux'], label='spectra')
    #plt.plot(fit['waveobs'], fit['flux'], label='fit')
    #plt.legend(loc='best')
    #plt.xlim([650,670])
    #plt.show()
    
    a = abund_line_by_line(my_random_star, param, c_model,what_object,"grid")
    
    #abd = abund_line_by_line(star_spectrum, params, code="grid")
    
    return param 
    
    
    

#execute calculations

if __name__ == '__main__':
    num_cores = 4
    p = Pool(num_cores)
    p.map(do_stuff, obj_id[0:4])

########## you can use it one by one in interactive mode in jupyter###########

#  what_object=obj_id[1]
#  degrade_res=47000
#  files_for_obj = get_files_for_obj_id(what_object, obj_id, fits_path)
#  files_sorted = read_wave_and_return_sorted(files_for_obj)
#  my_random_star = get_merged_spectra(files_sorted) #star i selected
#  plt.plot(my_random_star['waveobs'],my_random_star['flux'])
#  my_random_star, bv = clean_telluric_regions(my_random_star) #my clean star
#  print('---------- Rv from teluric -------------')
#  print(bv)
#  r_v, _ = determine_radial_velocity_with_template(my_random_star)
#  print('---------- Rv from template -------------')
#  print(r_v)
#  my_random_star = correct_radial_velocity(my_random_star, r_v)
#  plt.plot(my_random_star['waveobs'],my_random_star['flux'])
#  plt.ylim([0,0.35])
#      
#  
#  
#  # In[59]:
#  
#  
#  print('---------- SNR from flux -------------')
#  print(estimate_snr_from_flux(my_random_star))
#  print('---------- SNR from err -------------')
#  print(estimate_snr_from_err(my_random_star))
#  print('---------- Initial spectra resolution ---------')
#  myres = fits.open(files_for_obj[0])[0].header['SPEC_RES']
#  print(myres)
#  print('---------- Fit continuum and normalize ---------')
#  my_random_star, c_model = normalize_spectrum(my_random_star, myres)
#  ispec.write_spectrum(my_random_star, "./normalized.fits")
#  plt.plot(my_random_star['waveobs'],my_random_star['flux'])
#  plt.ylim([0,1.1])
#  
#  
#  # In[62]:
#  
#  
#  print('---------- Determine parameters using grid ---------')
#  param, err, fit = param_using_grid(my_random_star, c_model, what_object, myres)
#  print(param)
#  print(err)
#  plt.plot(my_random_star['waveobs'], my_random_star['flux'], label='spectra')
#  plt.plot(fit['waveobs'], fit['flux'], label='fit')
#  plt.legend(loc='best')
#  plt.xlim([515,525])
#  plt.ylim([0,1.1])
#  plt.show()
#  
#  
#  # In[61]:
#  
#  
#  a = abund_line_by_line(my_random_star, param, c_model,what_object,"grid")
#  #print(param)
#  
#  
#  # In[64]:
#  
#  
#  plt.plot(my_random_star['waveobs'], my_random_star['flux'], label='spectra')
#  plt.plot(fit['waveobs'], fit['flux'], label='fit')
#  plt.legend(loc='best')
#  plt.xlim([515,520])
#  plt.ylim([0,1.1])
#  plt.show()
#  #p = Pool(4)
#  #p.map(do_stuff, obj_id[1:5])
#  
