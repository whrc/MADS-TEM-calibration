#STEP3-CR
# parameters: nmax,krb(leaf,stem,root)
# targets: GPP, NPPAll, VegCarbon

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

    return dvmdostem.get_calibration_outputs()[16:32] #GPP/NPP

def get_param_targets():
    return dvmdostem.get_calibration_outputs(calib=True)[16:32] #GPP/NPP

dvmdostem=TEM.TEM_model()
dvmdostem.calib_mode='VEGC' 
dvmdostem.opt_run_setup='--pr-yrs 100 --eq-yrs 200 --sp-yrs 0 --tr-yrs 0 --sc-yrs 0'
dvmdostem.set_params(cmtnum=5, params=['nmax','nmax','nmax','nmax','nmax','nmax','nmax','nmax', \
				       'krb(0)','krb(0)','krb(0)','krb(0)','krb(0)','krb(0)','krb(0)','krb(0)', \
                                       'krb(1)','krb(1)','krb(1)',  \
                                       'krb(2)','krb(2)','krb(2)','krb(2)','krb(2)' ], \
                               pftnums=[0,1,2,3,4,5,6,7, \
					0,1,2,3,4,5,6,7, \
                                        0,1,2, \
                                        0,1,2,3,4])
"""
initial_guess=[18.515, 55.92, 6.786, 2.112, 39.6895, 3.0, 3.0, 3.0,
	       -0.111603, -0.112325, -0.100283, -13.52455, -3.833425, -9.983969, -0.62476, -3.503177, 
               -9.546573, -3.540054, -4.293397,
               -0.679503, -3.183039,  -6.802387, -0.104514, -6.813752]
y_init=PyCall.py"run_TEM"(initial_guess)

function TEM_pycall(parameters::AbstractVector)
        predictions = PyCall.py"run_TEM"(parameters)
        return predictions
end
obs=PyCall.py"get_param_targets"()
obs_time=1:length(obs)

md = Mads.createproblem(initial_guess, obs, TEM_pycall;
    paramkey=["nmax0","nmax1","nmax2","nmax3","nmax4","nmax5","nmax6","nmax7",
	      "krb00","krb01","krb02","krb03","krb04","krb05","krb06","krb07",
              "krb10","krb11","krb12",
              "krb20","krb21","krb22","krb23","krb24"],
    paramdist=["Uniform(1, 60)","Uniform(1, 60)","Uniform(1, 60)","Uniform(1, 60)",
               "Uniform(1, 60)","Uniform(1, 60)","Uniform(1, 60)","Uniform(1, 60)",
        "Uniform(-10, -0.1)","Uniform(-10, -0.1)","Uniform(-10, -0.1)","Uniform(-15, -0.1)","Uniform(-10, -0.1)",
        "Uniform(-10, -0.1)","Uniform(-10, -0.1)","Uniform(-10, -0.1)","Uniform(-10, -0.1)","Uniform(-10, -0.1)",
        "Uniform(-10, -0.1)","Uniform(-10, -0.1)","Uniform(-10, -0.1)","Uniform(-10, -0.1)","Uniform(-10, -0.1)",
        "Uniform(-10, -0.1)"],
    obstime=obs_time,
    #obsweight=[100,10,10,10,10,10,10,10,90,100,50,10,10,10,50,100,100,100,50,10,10,50,100,100],
    problemname="Calibration_STEP3-CR")

Mads.showparameters(md)
Mads.showobservations(md)


calib_random_results = Mads.calibraterandom(md, 3; seed=2021, all=true, tolOF=0.01, tolOFcount=4)

calib_random_estimates = hcat(map(i->collect(values(calib_random_results[i,3])), 1:3)...)

forward_predictions = Mads.forward(md, calib_random_estimates)
Mads.spaghettiplot(md, forward_predictions,
                       xtitle="# of observations", ytitle="GPP/NPP(Step3,CR)",filename="STEP3_CR_matchplot.png")
