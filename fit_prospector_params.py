import emcee
import dynesty
import time, sys, os
import h5py
import numpy as np
import scipy
import sedpy
import prospect
import fsps
from mpi4py import MPI
from astropy.io import fits
from prospect.utils.obsutils import fix_obs
from prospect.models import priors
from prospect.io import write_results as writer
from prospect.io import read_results as reader
from prospect import prospect_args
from prospect.fitting import fit_model
from prospect.models.templates import TemplateLibrary
from prospect.models.sedmodel import SedModel
from prospect.models import priors
from prospect.likelihood import lnlike_spec, lnlike_phot, write_log
from prospect.likelihood import chi_spec, chi_phot
from prospect.fitting import lnprobfn
from prospect.sources import CSPSpecBasis
from matplotlib.pyplot import *
from matplotlib.font_manager import FontProperties
from matplotlib import gridspec
from astropy.table import Table

path = '/data/groups/leauthaud/yluo42/prospector/lux/sed_fitting/pros_fit/'
phots_table = Table.read(path+'cosmos_HSC_Ha_OIII_v1.2.fits')
galaxies_number = len(phots_table)

def build_obs(snr=50,mags=np.array({}),**extras):
    obs = {}
    BB = ['hsc_{0}'.format(b) for b in ['g','r','i','z','y']]
    NB = ['{0}'.format(b) for b in ['N708','N540']]
    filternames = BB + NB
    obs["filters"] = sedpy.observate.load_filters(filternames)
    obs["maggies"] = 10**(-0.4*mags)
    obs["maggies_unc"] = (1./snr) * obs["maggies"]
    obs["phot_wave"] = np.array([f.wave_effective for f in obs["filters"]])
    obs["wavelength"] = None
    obs["spectrum"] = None
    obs['unc'] = None
    obs['mask'] = None
    obs = fix_obs(obs)
                                      
    return obs

def build_model(object_redshift=None,ldist=None,fixed_metallicity=None,add_duste=None,
                add_nebular=True,add_burst=True,**extras):
    model_params = TemplateLibrary["parametric_sfh"]
    model_params["tau"]["init"] = 3.0
    model_params["dust2"]["init"] = 0.1
    model_params["logzsol"]["init"] = -0.8
    model_params["tage"]["init"] = 6.5
    model_params["mass"]["init"] = 1e8
    model_params["tage"]["prior"] = priors.TopHat(mini=0.1, maxi=11.)
    model_params["dust2"]["prior"] = priors.TopHat(mini=0.0, maxi=2.0)
    model_params["tau"]["prior"] = priors.LogUniform(mini=1., maxi=7.)
    model_params["mass"]["prior"] = priors.LogUniform(mini=1e7, maxi=1e12)
    model_params["zred"]["prior"] = priors.TopHat(mini=0.005, maxi=4.0)
    model_params["mass"]["disp_floor"] = 1e5
    model_params["tau"]["disp_floor"] = 1.0
    model_params["tage"]["disp_floor"] = 1.0
    
    if fixed_metallicity is not None:
        model_params["logzsol"]["isfree"] = False
        model_params["logzsol"]['init'] = fixed_metallicity
    if object_redshift is not None:
        model_params["zred"]['isfree'] = False
        model_params["zred"]['init'] = object_redshift
    if object_redshift is None:
        model_params["zred"]['isfree'] = True
        model_params["zred"]['init'] = 0.1
    if add_duste:
        model_params.update(TemplateLibrary["dust_emission"])
    if add_nebular:
        model_params.update(TemplateLibrary["nebular"])
    if add_burst:
        model_params.update(TemplateLibrary["burst_sfh"])

    model_params["mass"]["isfree"] = True
    model_params["logzsol"]["isfree"] = True
    model_params["dust2"]["isfree"] = True
    model_params["tage"]["isfree"] = True
    model_params["tau"]["isfree"] = True
    model = SedModel(model_params)

    return model

def build_sps(zcontinuous=1, **extras):
    sps = CSPSpecBasis(zcontinuous=zcontinuous)
    return sps

def run_SED_fitting(galaxy_ID):
    table_ID = phots_table['ID'][galaxy_ID]
    print('start run for galaxy No.',table_ID)
    # Initialize the parameters
    output = {}
    run_params = {}
    run_params["mags"] = phots_table[galaxy_ID][0:7]
    run_params["snr"] = 50.0
    run_params["object_redshift"] = None
    run_params["fixed_metallicity"] = None
    run_params["add_duste"] = None
    run_params["add_nebular"] = True
    run_params["add_burst"] = None
    run_params["zcontinuous"] = 1
    # Here we will run all our building functions
    obs = build_obs(**run_params)
    sps = build_sps(**run_params)
    model = build_model(**run_params)
    # Generate the model SED at the initial value of theta
    theta = model.theta.copy()
    initial_spec, initial_phot, initial_mfrac = model.sed(theta, obs=obs, sps=sps)
    print("Done initialization")
    # --- run dynesty ----
    run_params["dynesty"] = True
    run_params["optmization"] = False
    run_params["emcee"] = False
    run_params["nested_method"] = "rwalk"
    run_params["nlive_init"] = 100
    run_params["nlive_batch"] = 100
    run_params["nested_dlogz_init"] = 0.05
    run_params["nested_posterior_thresh"] = 0.05
    run_params["nested_maxcall"] = int(1e5)
    output = fit_model(obs, model, sps, lnprobfn=lnprobfn, **run_params)
    print('Dynesty finished')
    # --- print some useful result ----
    imax = np.argmax(output["sampling"][0]['logl'] + model.prior_product(output["sampling"][0]['samples']))
    theta_max = output["sampling"][0]['samples'][imax, :]
    mspec_map, mphot_map, mextra_map = model.mean_model(theta_max, obs, sps=sps)
    z_fit_dynesty = theta_max[0]
    mass_fit_dynesty = np.log10(theta_max[1])
    logzsol = theta_max[2]
    dust2 = theta_max[3]
    tage = theta_max[4]
    tau = theta_max[5]
    mass_fit_dynesty_norm = np.log10(theta_max[1]*mextra_map)
    z_cosmos = phots_table['z_cosmos'][galaxy_ID]
    mass_cosmos = phots_table['mass_cosmos'][galaxy_ID]
    RA = phots_table['RA'][galaxy_ID]
    DEC = phots_table['DEC'][galaxy_ID]

    return RA, DEC, z_cosmos, mass_cosmos, z_fit_dynesty, mass_fit_dynesty, logzsol, dust2, tage, tau, mextra_map, mass_fit_dynesty_norm

f=open(path+"output_pros_fit.txt","w")
def run_mpi_fit(galaxy_ID):
    table_ID = phots_table['ID'][galaxy_ID]
    RA, DEC, z_cosmos, mass_cosmos, z_fit_dynesty, mass_fit_dynesty, logzsol, dust2, tage, tau, mextra_map, mass_fit_dynesty_norm = run_SED_fitting(galaxy_ID)
    output_results = str(table_ID)+' '+str(RA)+' '+str(DEC)+' '+str(z_cosmos)+' '+str(mass_cosmos)+' '+str(mass_fit_dynesty_norm)+' '+str(z_fit_dynesty)+' '+str(mass_fit_dynesty)+' '+str(logzsol)+' '+str(dust2)+' '+str(tage)+' '+str(tau)+' '+str(mextra_map)+'\n'
    f=open(path+"output_pros_fit.txt","a")
    f.write(output_results)
    f.close()

galaxy_number = galaxies_number
task_list = range(galaxy_number)

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

for i,task in enumerate(task_list):
    if i%size!=rank: continue
    print("Task number %d (%d) being done by processor %d of %d" % (i, task, rank, size))
    run_mpi_fit(task)
