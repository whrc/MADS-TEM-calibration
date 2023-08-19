# Use this .sh script to run an experiment in the deep soil branch.
# be sure to use the correct folder names and paths 
# Once runs are complete, it you'd like to have depth interpolated output, use Output_Synthesis_Valeria.sh (this will call 
# Layer_var_synth.py)

#CONTROL CASE MD1
mkdir -p /data/workflows/MD_deep_soil/control_allout_monthly_11px

./scripts/setup_working_directory.py /data/workflows/MD_deep_soil/control_allout_monthly_11px --input-data-path /data/input-catalog/cru-ts40_ar5_rcp85_mri-cgcm3_MurphyDome_10x10
./scripts/runmask-util.py --reset /data/workflows/MD_deep_soil/control_allout_monthly_11px/run-mask.nc
./scripts/runmask-util.py --yx 1 1 /data/workflows/MD_deep_soil/control_allout_monthly_11px/run-mask.nc

outspec_utils.py config/output_spec.csv --on ALD m
outspec_utils.py config/output_spec.csv --on DEEPC m
outspec_utils.py config/output_spec.csv --on DEEPDZ m
outspec_utils.py config/output_spec.csv --on EET m
outspec_utils.py config/output_spec.csv --on PET m
outspec_utils.py config/output_spec.csv --on HKDEEP m layer
outspec_utils.py config/output_spec.csv --on HKLAYER m layer
outspec_utils.py config/output_spec.csv --on HKMINEA m layer
outspec_utils.py config/output_spec.csv --on HKMINEB m layer
outspec_utils.py config/output_spec.csv --on HKMINEC m layer
outspec_utils.py config/output_spec.csv --on HKSHLW m layer
outspec_utils.py config/output_spec.csv --on IWCLAYER m layer
outspec_utils.py config/output_spec.csv --on LAYERDZ m layer
outspec_utils.py config/output_spec.csv --on LAYERTYPE m layer
outspec_utils.py config/output_spec.csv --on LAYERDEPTH m layer
outspec_utils.py config/output_spec.csv --on LWCLAYER m layer
outspec_utils.py config/output_spec.csv --on MINEC m layer
outspec_utils.py config/output_spec.csv --on LWCLAYER m layer
outspec_utils.py config/output_spec.csv --on SHLWC m layer
outspec_utils.py config/output_spec.csv --on SHLWDZ m layer
outspec_utils.py config/output_spec.csv --on TCDEEP m layer
outspec_utils.py config/output_spec.csv --on TCLAYER m layer
outspec_utils.py config/output_spec.csv --on TCMINEA m layer
outspec_utils.py config/output_spec.csv --on TCMINEB m layer
outspec_utils.py config/output_spec.csv --on TCMINEC m layer
outspec_utils.py config/output_spec.csv --on TCSHLW m layer
outspec_utils.py config/output_spec.csv --on RH m
outspec_utils.py config/output_spec.csv --on TDEEP m layer
outspec_utils.py config/output_spec.csv --on TLAYER m layer
outspec_utils.py config/output_spec.csv --on TMINEA m layer
outspec_utils.py config/output_spec.csv --on TMINEB m layer
outspec_utils.py config/output_spec.csv --on TMINEC m layer
outspec_utils.py config/output_spec.csv --on VWCLAYER m layer
outspec_utils.py config/output_spec.csv --on VWCMINEA m layer
outspec_utils.py config/output_spec.csv --on VWCMINEB m layer
outspec_utils.py config/output_spec.csv --on VWCMINEC m layer
outspec_utils.py config/output_spec.csv --on VWCSHLW m layer
outspec_utils.py config/output_spec.csv --on WATERTAB m layer
outspec_utils.py config/output_spec.csv --on QDRAINAGE m layer
outspec_utils.py config/output_spec.csv --on QDRAINLAYER m layer
outspec_utils.py config/output_spec.csv --on QINFILTRATION m layer
outspec_utils.py config/output_spec.csv --on QRUNOFF m layer
outspec_utils.py config/output_spec.csv --on LTRFALC m layer
outspec_utils.py config/output_spec.csv --on RAINFALL m layer
outspec_utils.py config/output_spec.csv --on SNOWFALL m layer


# cp -r /work/deep_soil_runs/output_spec.csv /data/workflows/MD_deep_soil/control_allout_monthly_11px/config/
outspec_utils.py config/output_spec.csv --summary

cd /data/workflows/MD_deep_soil/control_allout_monthly_11px

/work/dvmdostem -l err -f /data/workflows/MD_deep_soil/control_allout_monthly_11px/config/config.js --force-cmt 1 -p 100 -e 1000 -s 250 -t 115 -n 85

cd /work/deep_soil

vi Output_synthesis_Valeria.sh

bash Output_synthesis_Valeria.sh

# use these years for each step: pr 100 eq 1000 sp 250 tr 115 sc 85

##################################################################################################################################
#CONTROL CASE BNZ1
mkdir -p /data/workflows/BNZ_deep_soil/control_allout_monthly_11px_updSoil

./scripts/setup_working_directory.py /data/workflows/BNZ_deep_soil/control_allout_monthly_11px_updSoil --input-data-path /data/input-catalog/cru-ts40_ar5_rcp85_ncar-ccsm4_bonanzacreeklter_10x10
./scripts/runmask-util.py --reset /data/workflows/BNZ_deep_soil/control_allout_monthly_11px_updSoil/run-mask.nc
./scripts/runmask-util.py --yx 1 1 /data/workflows/BNZ_deep_soil/control_allout_monthly_11px_updSoil/run-mask.nc

outspec_utils.py config/output_spec.csv --summary

cd /data/workflows/BNZ_deep_soil/control_allout_monthly_11px_updSoil

/work/dvmdostem -l err -f /data/workflows/BNZ_deep_soil/control_allout_monthly_11px_updSoil/config/config.js --force-cmt 1 -p 100 -e 1000 -s 250 -t 115 -n 85


##################################################################################################################################
#DEEP_SOIL RUN MD1
git checkout deep_soil
# use the same code as the control cases, but make sure checked out deep_soil branch first

mkdir -p /data/workflows/MD_deep_soil/nfactorw0.75_hk02_monthly_11px

./scripts/setup_working_directory.py /data/workflows/MD_deep_soil/nfactorw0.75_hk02_monthly_11px --input-data-path /data/input-catalog/cru-ts40_ar5_rcp85_mri-cgcm3_MurphyDome_10x10
./scripts/runmask-util.py --reset /data/workflows/MD_deep_soil/nfactorw0.75_hk02_monthly_11px/run-mask.nc
./scripts/runmask-util.py --yx 1 1 /data/workflows/MD_deep_soil/nfactorw0.75_hk02_monthly_11px/run-mask.nc

# outspec_utils.py config/output_spec.csv --on ALD y
# outspec_utils.py config/output_spec.csv --on DEEPC y
# outspec_utils.py config/output_spec.csv --on DEEPDZ y
# outspec_utils.py config/output_spec.csv --on EET y
# outspec_utils.py config/output_spec.csv --on PET y
# outspec_utils.py config/output_spec.csv --on HKDEEP y layer
# outspec_utils.py config/output_spec.csv --on HKLAYER y layer
# outspec_utils.py config/output_spec.csv --on HKMINEA y layer
# outspec_utils.py config/output_spec.csv --on HKMINEB y layer
# outspec_utils.py config/output_spec.csv --on HKMINEC y layer
# outspec_utils.py config/output_spec.csv --on HKSHLW y layer
# outspec_utils.py config/output_spec.csv --on IWCLAYER y layer
# outspec_utils.py config/output_spec.csv --on LAYERDZ y layer
# outspec_utils.py config/output_spec.csv --on LAYERDEPTH y layer
# outspec_utils.py config/output_spec.csv --on LWCLAYER y layer
# outspec_utils.py config/output_spec.csv --on MINEC y layer
# outspec_utils.py config/output_spec.csv --on LWCLAYER y layer
# outspec_utils.py config/output_spec.csv --on SHLWC y layer
# outspec_utils.py config/output_spec.csv --on SHLWDZ y layer
# outspec_utils.py config/output_spec.csv --on TCDEEP y layer
# outspec_utils.py config/output_spec.csv --on TCLAYER y layer
# outspec_utils.py config/output_spec.csv --on TCMINEA y layer
# outspec_utils.py config/output_spec.csv --on TCMINEB y layer
# outspec_utils.py config/output_spec.csv --on TCMINEC y layer
# outspec_utils.py config/output_spec.csv --on TCSHLW y layer
# outspec_utils.py config/output_spec.csv --on RH y
# outspec_utils.py config/output_spec.csv --on TDEEP y layer
# outspec_utils.py config/output_spec.csv --on TLAYER y layer
# outspec_utils.py config/output_spec.csv --on TMINEA y layer
# outspec_utils.py config/output_spec.csv --on TMINEB y layer
# outspec_utils.py config/output_spec.csv --on TMINEC y layer
# outspec_utils.py config/output_spec.csv --on VWCLAYER y layer
# outspec_utils.py config/output_spec.csv --on VWCMINEA y layer
# outspec_utils.py config/output_spec.csv --on VWCMINEB y layer
# outspec_utils.py config/output_spec.csv --on VWCMINEC y layer
# outspec_utils.py config/output_spec.csv --on VWCSHLW y layer
# outspec_utils.py config/output_spec.csv --on WATERTAB y layer

cd /data/workflows/MD_deep_soil/nfactorw0.75_hk02_monthly_11px

/work/dvmdostem -l err -f /data/workflows/MD_deep_soil/nfactorw0.75_hk02_monthly_11px/config/config.js --force-cmt 1 -p 100 -e 1000 -s 250 -t 115 -n 85

##################################################################################################################################
#DEEP_SOIL RUN BNZ1

git checkout deep_soil
# use the same code as the control cases, but make sure checked out deep_soil branch first, and use $make clean and $make

mkdir -p /data/workflows/BNZ_deep_soil/experiment_allout_monthly_11px_updSoil

./scripts/setup_working_directory.py /data/workflows/BNZ_deep_soil/experiment_allout_monthly_11px_updSoil --input-data-path /data/input-catalog/cru-ts40_ar5_rcp85_ncar-ccsm4_bonanzacreeklter_10x10
./scripts/runmask-util.py --reset /data/workflows/BNZ_deep_soil/experiment_allout_monthly_11px_updSoil/run-mask.nc
./scripts/runmask-util.py --yx 1 1 /data/workflows/BNZ_deep_soil/experiment_allout_monthly_11px_updSoil/run-mask.nc

outspec_utils.py config/output_spec.csv --summary

cd /data/workflows/BNZ_deep_soil/experiment_allout_monthly_11px_updSoil

/work/dvmdostem -l err -f /data/workflows/BNZ_deep_soil/experiment_allout_monthly_11px_updSoil/config/config.js --force-cmt 1 -p 100 -e 1000 -s 250 -t 115 -n 85

