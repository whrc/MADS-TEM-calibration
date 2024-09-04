# STEP3 MD1-R
#       MD1-R-1 updated IG for OF 108K
# parameters: krb
# targets: VegCarbon(Leaf,Stem,Root) [12:22]

import Mads
import PyCall
include("write_csv.jl")
@show pwd()

PyCall.py"""

import sys,os
sys.path.append(os.path.join('/work','scripts'))
import TEM

def run_TEM(x):
    
    for j in range(len(dvmdostem.params)):
        dvmdostem.params[j]['val']=x[j]   
    # update param files
    dvmdostem.clean()
    dvmdostem.setup(calib=True)
    dvmdostem.update_params()
    dvmdostem.run()

    return dvmdostem.get_calibration_outputs()[12:22] #VEGC

def get_param_targets():
    return dvmdostem.get_calibration_outputs(calib=True)[12:22] #VEGC

dvmdostem=TEM.TEM_model()
dvmdostem.site='/data/input-catalog/cru-ts40_ar5_rcp85_mri-cgcm3_MurphyDome_10x10'
dvmdostem.work_dir='/data/workflows/MD1-R'
dvmdostem.calib_mode='VEGC' 
dvmdostem.opt_run_setup='--pr-yrs 100 --eq-yrs 200 --sp-yrs 0 --tr-yrs 0 --sc-yrs 0'
dvmdostem.set_params(cmtnum=1, params=['krb(0)','krb(0)','krb(0)','krb(0)', \
                                       'krb(1)','krb(1)','krb(1)',  \
                                       'krb(2)','krb(2)','krb(2)' ], \
                               pftnums=[0,1,2,3, \
                                        0,1,2, \
                                        0,1,2])
"""
initial_guess=[ -13.34, -17.42, -12.0, -4.83, 
	        -4.95, -4.7, -5.66, -13.04, -6.38,  -4.678]

	      # -6.0, -3.45, -2.95, -4.65,
	      # -4.88, -5.15, -6.65,
	      # -8.2, -6.2, -3.2]
y_init=PyCall.py"run_TEM"(initial_guess)
#print(y_init)
function TEM_pycall(parameters::AbstractVector)
        predictions = PyCall.py"run_TEM"(parameters)
        return predictions
end
obs=PyCall.py"get_param_targets"()
obs_time=1:length(obs)
#print(obs)

md = Mads.createproblem(initial_guess, obs, TEM_pycall;
    paramkey=["krb00","krb01","krb02","krb03",
              "krb10","krb11","krb12",
              "krb20","krb21","krb22"],
    paramdist=[
        "Uniform(-20, -0.1)","Uniform(-20, -0.1)","Uniform(-20, -0.1)","Uniform(-20, -0.1)",
        "Uniform(-20, -0.1)","Uniform(-20, -0.1)","Uniform(-20, -0.1)",
        "Uniform(-20, -0.1)","Uniform(-20, -0.1)","Uniform(-20, -0.1)"],
    obstime=obs_time,
    #obsweight=[100,25,100,25,25,25,25,25,25,25],
    problemname="STEP3-MD1-R-1-EJ")

Mads.showparameters(md)
Mads.showobservations(md)

calib_random_results = Mads.calibraterandom(md, 10; seed=2023, all=true, tolOF=0.01, tolOFcount=4)

calib_random_estimates = hcat(map(i->collect(values(calib_random_results[i,3])), 1:10)...)

forward_predictions = Mads.forward(md, calib_random_estimates)
Mads.spaghettiplot(md, forward_predictions,
                       xtitle="# of observations", ytitle="VEGC",filename="STEP3-MD1-R-1-EJ-matchplot.png")

save_csv(Mads.getparamkeys(md), Mads.getparamsmin(md), Mads.getparamsmax(md), initial_guess,
    Mads.getmadsrootname(md), Mads.getobsweight(md))

save_model_csv(md,Mads.getmadsrootname(md),forward_predictions)

