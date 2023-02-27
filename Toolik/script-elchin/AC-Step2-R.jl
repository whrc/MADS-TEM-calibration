import Mads
import PyCall
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

    return dvmdostem.get_calibration_outputs()[16:24]

def get_param_targets():
    return dvmdostem.get_calibration_outputs(calib=True)[16:24]

dvmdostem=TEM.TEM_model()
dvmdostem.calib_mode='NPPAll'
dvmdostem.opt_run_setup='--pr-yrs 100 --eq-yrs 200 --sp-yrs 0 --tr-yrs 0 --sc-yrs 0'
dvmdostem.set_params(cmtnum=5, params=['nmax','nmax','nmax','nmax','nmax','nmax','nmax','nmax'], \
                               pftnums=[0,1,2,3,4,5,6,7])
"""
initial_guess=[1.69, 17.28, 2.36, 2.02, 19.95, 3.0, 3.0, 3.0]
#initial_guess=[15.42, 38.58, 30.27, 1.96, 4.07, 20.79, 36.75, 14.91]
#[7.0, 8.2, 9.0, 27.6, 4.5, 3.0, 3.0, 3.0]
y_init=PyCall.py"run_TEM"(initial_guess)

function TEM_pycall(parameters::AbstractVector)
        predictions = PyCall.py"run_TEM"(parameters)
        return predictions
end
obs=PyCall.py"get_param_targets"()
obs_time=1:length(obs)

md = Mads.createproblem(initial_guess, obs, TEM_pycall;
    paramkey=["nmax0","nmax1","nmax2","nmax3","nmax4","nmax5","nmax6","nmax7"],
    paramdist=["Uniform(1, 60)","Uniform(1, 60)","Uniform(1, 60)","Uniform(1, 60)",
               "Uniform(1, 60)","Uniform(1, 60)","Uniform(1, 60)","Uniform(1, 60)"],
    obstime=obs_time,
    #obsweight=[100,10,100,100,10,10,100,100],
    problemname="Calibration_STEP2-R4")

Mads.showparameters(md)
Mads.showobservations(md)

calib_param, calib_information = Mads.calibrate(md, tolOF=0.01, tolOFcount=4)

calib_random_results = Mads.calibraterandom(md, 3; seed=2021, all=true, tolOF=0.01, tolOFcount=4)

calib_random_estimates = hcat(map(i->collect(values(calib_random_results[i,3])), 1:3)...)

forward_predictions = Mads.forward(md, calib_random_estimates)
Mads.spaghettiplot(md, forward_predictions,
                       xtitle="# of observations", ytitle="GPP(Step2,R4)",filename="STEP2_R4_matchplot.png")
