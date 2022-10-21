import numpy as np
from matplotlib import pyplot as plt

from pysted import base, utils
from pysted import exp_data_gen as dg
import copy


"""
This script will go over the basics of pySTED for simulation of confocal and STED 
acquisitions on simulated samples. In order to simulate an acquisition, we need a 
microscope and a sample. To build the STED microscope, we need an excitation beam, a STED 
beam, a detector and the parameters of the fluorophores used in the sample. The class 
code for the objects that make up the microscope and the sample are contained in 
pysted.base Each object has parameters which can be tuned, which will affect the 
resulting acquisition
"""

print("Setting up the microscope...")
# Fluorophore properties
FLUOs_dict = dict( 

	star635 = { # WARNING: bleaching dynamics for ATTO647N, sigma_ste and tau_vib and for ATTO647N, tau_tri for egfp
	    "lambda_": 655e-9, #https://abberior.shop/abberior-STAR-635
	    "qy": 0.88, #https://abberior.shop/abberior-STAR-635
	    "sigma_abs": {
	        635: 4.21e-20, # 0.94208 a.u,  EC=110000, https://abberior.shop/abberior-STAR-635
	        750: 2.81e-23,   # 0.00063 a.u, https://abberior.shop/abberior-STAR-635
	    },
	    "sigma_ste": {
	        750: 4.8e-22, #Table S3, Oracz2017
	    },
	    "tau": 2.8e-9,
	    "tau_vib": 1.0e-12, #t_vib, Table S3, Oracz2017 
	    "tau_tri": 25e-6, # pasted from egfp
	    "k0": 0, #Table S3, Oracz2017
	    "k1": 1.3e-15, #Table S3,  (changed seemingly wrong unit: 5.2 × 10−10 / (100**2)**1.4)
	    "b":1.4, #Table S3, Oracz2017
	    "triplet_dynamics_frac": 0, #Ignore the triplet dynamics by default
	},

	egfp = { # WARNING: bleaching dynamics for ATTO647N, tau_vib for GFP(wt) 
	    "lambda_": 507e-9, #EM lambda https://www.fpbase.org/protein/egfp/
	    "qy": 0.6, #https://www.microscopyu.com/techniques/fluorescence/introduction-to-fluorescent-proteins
	    "sigma_abs": {
	        488: 2.14e-20,   #0.9982 a.u, EC=55900 (https://www.fpbase.org/protein/egfp/)
	        575: 9.64e-23,   #0.0045 a.u (https://www.fpbase.org/protein/egfp/)
	    },
	    "sigma_ste": {
	        575: 1.5e-20, #Masters2018 (visual approximation on a plot data fitted curve)
# 	        575: 1.0e-21, #Beeson2015 ("Typical parameters for fluorescent organic molecules")
	    },
	    "tau": 2.6e-09, #https://www.fpbase.org/protein/egfp/
	    "tau_vib": 1.2e-12, # GFP(wt) Winkler2002 
	    "tau_tri": 25e-6, #Jimenez-Banzo2008
	    "k1": 1.3e-15, # Atto640N, Oracz2017
	    "b":1.4, # Atto640N, Oracz2017
	    "triplet_dynamics_frac": 0, #Ignore the triplet dynamics by default
	},
	ATTO647N = { #Warning tau_tri for egfp
	    "lambda_": 690e-9, # Figure 1, Oracz2017
	    "qy": 0.65,  # Product Information: ATTO 647N (ATTO-TEC GmbH)
	    "sigma_abs": {
	        635: 1.0e-20, #Table S3, Oracz2017
	        750: 3.5e-25,  # (1 photon exc abs) Table S3, Oracz2017
	    },
	    "sigma_ste": {
	        750: 4.8e-22, #Table S3, Oracz2017
	    },
	    "tau": 3.5e-9, #Table S3, Oracz2017
	    "tau_vib": 1.0e-12, #t_vib, Table S3, Oracz2017 
	    "tau_tri": 25e-6, # pasted from egfp
	    "k0": 0, #Table S3, Oracz2017
	    "k1": 1.3e-15, #Table S3,  (changed seemingly wrong unit: 5.2 × 10−10 / (100**2)**1.4)
	    "b":1.4, #Table S3, Oracz2017
	    "triplet_dynamics_frac": 0, #Ignore the triplet dynamics by default
	}
	)
	
FLUOs_dict["ATTO590"] = copy.deepcopy(FLUOs_dict["ATTO647N"])
FLUOs_dict["ATTO590"]["sigma_abs"][750] = 8e-26 #Table S4, Oracz2017
FLUOs_dict["ATTO590"]["sigma_ste"][750] = 4e-22 #Table S4, Oracz2017
FLUOs_dict["ATTO590"]["k0"] = 2.5e-5/100**2 #Table S4, Oracz2017 (changed seemingly wrong unit)
FLUOs_dict["ATTO590"]["k1"] = 9e-18/(100**2)**1.9 #Table S4, Oracz2017  (changed seemingly wrong unit)
FLUOs_dict["ATTO590"]["b"] = 1.9



pixelsize = 20e-9
# Generating objects necessary for acquisition simulation

fl = 'egfp'
# fl = 'ATTO647N'
# fl = 'ATTO590'
# det_delay = 750e-12
det_delay = 0
det_width = 8e-9
if fl == 'egfp':
	lambda_exc = 488e-9
	lambda_sted = 575e-9
	p_ex = 30 * 154e-6/100
	p_sted = 30 * 4.13e-3/100
elif fl=='ATTO647N' or fl=='ATTO590':
	lambda_exc = 635e-9
	lambda_sted = 750e-9
	p_ex = 3e-6 
	p_sted = 15e-3
fluo = base.Fluorescence(**FLUOs_dict[fl])
laser_ex = base.GaussianBeam(lambda_exc)
laser_sted = base.DonutBeam(lambda_sted, zero_residual=0.01, rate=40e6, tau=400e-12, \
							anti_stoke=False)  #Similar to the labs microscope
detector = base.Detector(noise=True, det_delay=det_delay, det_width=det_width, background=0) #Similar to the labs microscope
objective = base.Objective()



# # These are the parameter ranges our RL agents can select from when playing actions
# action_spaces = {
#     "p_sted" : {"low" : 0., "high" : 175e-3}, # Similar to the 775nm laser in our lab
#     "p_ex" : {"low" : 0., "high" : 150e-6}, # Similar to the 640nm laser in our lab
#     "pdt" : {"low" : 10.0e-6, "high" : 150.0e-6},
# }


# Example values of parameters used when doing a Confocal acquisition. Confocals always 
# have p_sted = 0
conf_params = {
    "pdt": 20e-6,
    "p_ex": p_ex,
    "p_sted": 0.   # params have to be floats to pass the C function
}

# Example values of parameters used when doing a STED acquisition
sted_params = copy.deepcopy(conf_params)
sted_params["p_sted"] = p_sted



# generate the microscope from its constituent parts
# if load_cache is true, it will load the previously generated microscope. This can save 
# time if a microscope was previsously generated and used the same pixelsize we are using 
# now
microscope = base.Microscope(laser_ex, laser_sted, detector, objective, fluo, load_cache=True)
i_ex, i_sted, _ = microscope.cache(pixelsize, save_cache=True)
# psf_conf = microscope.get_effective(pixelsize, action_spaces["p_ex"]["high"], 0.0)
# psf_sted = microscope.get_effective(pixelsize, action_spaces["p_ex"]["high"], action_spaces["p_sted"]["high"] * 0.25)

# You can uncomment these lines to visualize the simulated excitation and STED beams, as 
# well as the detection PSFs when using certain excitation / STED power combinations
# fig, axes = plt.subplots(2, 2)
#
# axes[0, 0].imshow(i_ex)
# axes[0, 0].set_title(f"Excitation beam")
#
# axes[0, 1].imshow(i_sted)
# axes[0, 1].set_title(f"STED beam")
#
# axes[1, 0].imshow(psf_conf)
# axes[1, 0].set_title(f"Detection PSF in confocal modality")
#
# axes[1, 1].imshow(psf_sted)
# axes[1, 1].set_title(f"Detection PSF in STED modality")
#
# plt.tight_layout()
# plt.show()

# we now need a sample on to which to do our acquisition, which we call the datamap
# I will show how to build a simple datamap, along with a more complex one which includes 
# nanostructures and a temporal element
# First, we use the Synapse class in exp_data_gen to simulate a synapse-like structure and 
# add nanostructures to it You could use any integer-valued array as a Datamap
shroom1 = dg.Synapse(5, mode="mushroom", seed=42)

n_molecs_in_domain1, min_dist1 = 135, 50
shroom1.add_nanodomains(10, min_dist_nm=min_dist1, n_molecs_in_domain=n_molecs_in_domain1, valid_thickness=7)

# create the Datamap and set its region of interest
dmap = base.Datamap(shroom1.frame, pixelsize)
dmap.set_roi(i_ex, "max")

shroom2 = dg.Synapse(5, mode="mushroom", seed=42)
n_molecs_in_domain2, min_dist2 = 0, 50
shroom2.add_nanodomains(10, min_dist_nm=min_dist2, n_molecs_in_domain=n_molecs_in_domain2, valid_thickness=7)

# create a temporal Datamap which will also contain information on the positions of 
# nanodomains We create a temporal element by making the nanostructures flash
# We then set its temporal index to be at the flash peak
time_idx = 2
temp_dmap = base.TemporalSynapseDmap(shroom2.frame, pixelsize, shroom2)
temp_dmap.set_roi(i_ex, "max")
temp_dmap.create_t_stack_dmap(2000000)
temp_dmap.update_whole_datamap(time_idx)
temp_dmap.update_dicts({"flashes": time_idx})

# you can uncomment this code to see both datamaps, which should look similar
# fig, axes = plt.subplots(1, 2)
#
# axes[0].imshow(dmap.whole_datamap[dmap.roi])
# axes[0].set_title(f"Base Datamap")
#
# axes[1].imshow(temp_dmap.whole_datamap[temp_dmap.roi])
# axes[1].set_title(f"Datamap with temporal element")
#
# plt.show()

# uncomment this code to run through the flash
# for t in range(temp_dmap.flash_tstack.shape[0]):
#     temp_dmap.update_whole_datamap(t)
#     temp_dmap.update_dicts({"flashes": t})
#
#     plt.imshow(temp_dmap.whole_datamap[temp_dmap.roi])
#     plt.title(f"Time idx = {t}")
#     plt.show()

# Now let's show a confocal acquisition and a STED acquisition on the datamaps
# The returns are :
# (1) The acquired image signal
# (2) The bleached datamaps
# (3) The acquired intensity. This is only useful when working in a temporal exeperiment 
# setting, in which an acquisition could be interrupted by the flash happening through it.


conf_acq, conf_bleached, _ = microscope.get_signal_and_bleach(dmap, dmap.pixelsize, **conf_params,
                                                              bleach=True, update=True)
conf_acq2, conf_bleached2, _ = microscope.get_signal_and_bleach(dmap, dmap.pixelsize, **conf_params,
                                                              bleach=True, update=True)
datamap_1 = copy.deepcopy(temp_dmap.whole_datamap[temp_dmap.roi])
sted_acq, sted_bleached, _ = microscope.get_signal_and_bleach(temp_dmap, temp_dmap.pixelsize, **sted_params,
                                                              bleach=True, update=True)
sted_acq2, sted_bleached2, _ = microscope.get_signal_and_bleach(temp_dmap, temp_dmap.pixelsize, **sted_params,
                                                              bleach=True, update=True)
datamap_2 = copy.deepcopy(temp_dmap.whole_datamap[temp_dmap.roi])



fig, axes = plt.subplots(2, 3)

vmax = conf_acq.max()
im = axes[0,0].imshow(conf_acq, vmax=vmax)
axes[0,0].set_title(f"Confocal 1")
plt.colorbar(im, ax=axes[0,0])


im = axes[0,1].imshow(conf_acq2, vmax=vmax)
axes[0,1].set_title(f"Confocal 2")
plt.colorbar(im, ax=axes[0,1])


im = axes[0,2].imshow(datamap_1)
axes[0,2].set_title(f"dmap")
plt.colorbar(im, ax=axes[0,2])


vmax = sted_acq.max()
im = axes[1,0].imshow(sted_acq, vmax=vmax)
axes[1,0].set_title(f"STED 1")
plt.colorbar(im, ax=axes[1,0])


im = axes[1,1].imshow(sted_acq2, vmax=vmax)
axes[1,1].set_title(f"STED 2")
plt.colorbar(im, ax=axes[1,1])

im = axes[1,2].imshow(datamap_2)
axes[1,2].set_title(f"dmap")
plt.colorbar(im, ax=axes[1,2])


plt.suptitle("The four images where acquired sequentially. \nSame normalization on each row")

plt.show()

# I have set the bleaching to false in these acquisitions for speed. You can set it to 
# True to see its effects on the acquired signal and the datamaps. You can also of course 
# modify other parameters to see their effects on the acquired images. :)
