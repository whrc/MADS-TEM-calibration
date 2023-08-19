
#!/usr/local/bin/bash

# Author: Helene Genet, UAF
# Creation date: Jan. 25 2022
# Purpose: general script to synthesis TEM outputs for data analysis



### INFORMATION REQUIRED

# path to the directory of TEM input files used to produce the TEM outputs
inputdir="/data/input-catalog/cru-ts40_ar5_rcp85_mri-cgcm3_MurphyDome_10x10"
# path to the TEM raw output directory
#rawoutdir="/Users/helene/Helene/TEM/DVMDOSTEM/dvmdostem_workflows/winter_resp/control/output/"
rawoutdir="/data/workflows/MD_deep_soil/control_allout_monthly_11px/output/"
#tar -zxvf "${rawoutdir}.tar.gz"
# path to the directory containing the python scripts associated to this bash (i.e. Layer_var_synth.py)
scriptdir="/work/deep_soil_runs/"
# path to the directory to store output synthesis files
outdir="${rawoutdir}synthesis/"
# historical period starting and ending years
hist_start=1901
hist_end=2015
# projection period starting and ending years
proj_start=2016
proj_end=2100
# list of simulation scenarios of interest: pr, eq, sp, tr, sc for pre-run, equilibrium, spin-up, historical and scenario runs
# sclist=(tr sc)
sclist=(tr sc)
# name of the scenario run
scname=mri
# name of the variable of interest
# var=TLAYER
var=TCMINEA
# for variables with soil layer dimension, specify the soil depths, in meter, at which you woould like the variable to be synthesized
# depthlist=(0.10 0.20 0.50 1.00)
depthlist=(0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1
       0.11 0.12 0.13 0.14 0.15 0.16 0.17 0.18 0.19 0.2 0.21
       0.22 0.23 0.24 0.25 0.26 0.27 0.28 0.29 0.3 0.31 0.32
       0.33 0.34 0.35 0.36 0.37 0.38 0.39 0.4 0.41 0.42 0.43
       0.44 0.45 0.46 0.47 0.48 0.49 0.5 0.51 0.52 0.53 0.54
       0.55 0.56 0.57 0.58 0.59 0.6 0.61 0.62 0.63 0.64 0.65
       0.66 0.67 0.68 0.69 0.7 0.71 0.72 0.73 0.74 0.75 0.76
       0.77 0.78 0.79 0.8 0.81 0.82 0.83 0.84 0.85 0.86 0.87
       0.88 0.89 0.9 0.91 0.92 0.93 0.94 0.95 0.96 0.97 0.98
       0.99 1 1.01 1.02 1.03 1.04 1.05 1.06 1.07 1.08 1.09
       1.1 1.11 1.12 1.13 1.14 1.15 1.16 1.17 1.18 1.19 1.2 
       1.21 1.22 1.23 1.24 1.25 1.26 1.27 1.28 1.29 1.3 1.31
       1.32 1.33 1.34 1.35 1.36 1.37 1.38 1.39 1.4 1.41 1.42
       1.43 1.44 1.45 1.46 1.47 1.48 1.49 1.5 1.51 1.52 1.53
       1.54 1.55 1.56 1.57 1.58 1.59 1.6 1.61 1.62 1.63 1.64
       1.65 1.66 1.67 1.68 1.69 1.7 1.71 1.72 1.73 1.74 1.75
       1.76 1.77 1.78 1.79 1.8 1.81 1.82 1.83 1.84 1.85 1.86
       1.87 1.88 1.89 1.9 1.91 1.92 1.93 1.94 1.95 1.96 1.97
       1.98 1.99
       2.125 2.25 2.375 2.5 2.625 2.75 2.875 3
       3.125 3.25 3.375 3.5 3.625 3.75 3.875 4 4.125
       4.25 4.375 4.5 4.625 4.75 4.875 5)



### GETTING STARTED...

cd $rawoutdir
mkdir $outdir



### DETERMINE OUTPUTFILE TIME RESOLUTION AND LIST OF DIMENSIONS

## 1- Time resolution

# list the time frequency of all output files of interest
freq=()
for sc in "${sclist[@]}"; do
  #1- determine frequency of outputs
  if [[ $(basename "${rawoutdir}${var}_"*"_${sc}.nc") == *"monthly"* ]];then
    fq="monthly"
  elif [[ $(basename "${rawoutdir}${var}_"*"_${sc}.nc") == *"yearly"* ]];then
    fq="yearly";
  fi
  freq=("${freq[@]}" "${fq[@]}")
done
# check that the time resolution is the same for all output files.
echo ${freq[@]}
mapfile -t tres < <(printf "%s\n" "${freq[@]}" | sort -u)
if (( ${#tres[@]} > 1 ));then
  echo "All output files do not have the same time frequency. Please homogenize the time resolution of the output files. [To do so, you can rather reproduce the simulations with homogenous time resolution, or use the ncwa operator (nco) to rather sum (fluxes) or take the december value (stocks) of monthly outputs to generate yearly outputs which are the coarsest resolution of TEM outputs]."
else
  echo "The time resolution of the output files is: ${tres[@]}"
fi


## 2- Dimensions

# list the dimensions from all output files of interest and check they are all the same.
prevdimlist=()
for sc in "${sclist[@]}"; do
  echo $sc;
  ncdump -h "${rawoutdir}${var}_"*"_${sc}.nc" > ./a.txt
  IFS=' ' read -r -a dimlist <<< "`grep $var'(' a.txt | awk -F"[()]" '{print $2}' | sed 's/ //g' | sed 's/,/ /g'`"
  if (( ${#prevdimlist[@]} > 0 ));then
    diff=$(diff <(printf "%s\n" "${dimlist[@]}") <(printf "%s\n" "${prevdimlist[@]}"))
    if [[ ! -z "$diff" ]]; then
      echo "The list of dimensions is not the same in all output files of interest. Please homogenize. [To do so, you can rather reproduce the simulations with homogenous dimensions, or use the ncwa operator (nco) to sum or averag the variable values along the dimensions that are missing in some of the outputs files of interest]."
      dimlist=()
      break
    fi
  fi
  prevdimlist=("${dimlist[@]}")
done
echo "The list of dimensions in the output files is: ${dimlist[@]}"




### PRODUCE SOIL LAYER OUTPUTS OF COMPARABLE DEPTH


for sc in "${sclist[@]}"; do
  echo "scenario ${sc}"
  # make sure that layer dz and layer types are present
  if [[ ! -f "${rawoutdir}LAYERTYPE_${tres[@]}_${sc}.nc" ]];then
    echo "LAYERTYPE for ${sc} mode doesn't exist. Yet it is need to run this procedure. Break";
    break
  elif [[ ! -f "${rawoutdir}LAYERDZ_${tres[@]}_${sc}.nc" ]];then
    echo "LAYERDZ for ${sc} mode doesn't exist. Yet it is need to run this procedure. Break";
    break
  fi
  cp "${rawoutdir}LAYERTYPE_${tres[@]}_${sc}.nc" "${rawoutdir}SOILSTRUCTURE_${tres[@]}_${sc}.nc"
  ncks -A -h "${rawoutdir}LAYERDZ_${tres[@]}_${sc}.nc" "${rawoutdir}SOILSTRUCTURE_${tres[@]}_${sc}.nc"
  # export the necessary info to run the python code
  export scrun=$sc
  export inpath=$rawoutdir
  export outpath=$outdir
  export ncvar=$var
  export timeres=${tres[0]}
  export dmnl=${dimlist[@]}
  for i in "${dimlist[@]:1}"; do
    dmnl+=,$i
  done
  export dl=${depthlist[0]}
  for i in "${depthlist[@]:1}"; do
    dl+=,$i
  done
  #run the python script for linear interpolation
  python3 "${scriptdir}Layer_var_synth.py"
done

