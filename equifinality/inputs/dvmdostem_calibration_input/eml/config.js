{
  "general": {
    "run_name": "A sample dvmdostem run. Modify this text to suit your needs."
  },
  "IO": {
    "parameter_dir": "/data/workflows/WrPMIP/eml/parameters/",
    "hist_climate_file": "/data/input-catalog/eml/historic-climate.nc",
    "proj_climate_file": "/data/input-catalog/eml/projected-climate.nc",
    "veg_class_file": "/data/input-catalog/eml/vegetation.nc",
    "drainage_file": "/data/input-catalog/eml/drainage.nc",
    "soil_texture_file": "/data/input-catalog/eml/soil-texture.nc",
    "co2_file": "/data/input-catalog/eml/co2.nc",
    "proj_co2_file": "/data/input-catalog/eml/projected-co2.nc",
    "topo_file": "/data/input-catalog/eml/topo.nc",
    "fri_fire_file": "/data/input-catalog/eml/fri-fire.nc",
    "hist_exp_fire_file": "/data/input-catalog/eml/historic-explicit-fire.nc",
    "proj_exp_fire_file": "/data/input-catalog/eml/projected-explicit-fire.nc",
    "runmask_file": "/data/workflows/WrPMIP/eml/run-mask.nc",
    "output_dir": "/data/workflows/WrPMIP/eml/output/",
    "output_spec_file": "/data/workflows/WrPMIP/eml/config/output_spec.csv",
    "output_monthly": 1,
    "output_nc_eq": 0,
    "output_nc_sp": 0,
    "output_nc_tr": 1,
    "output_nc_sc": 1,
    "output_interval": 1
  },
  "calibration-IO": {
    "unique_pid_tag": "",
    "caldata_tree_loc": "/tmp/eml"
  },
  "stage_settings": {
    "restart_mode": "restart",
    "inter_stage_pause": false,
    "pr": {
      "env": true,
      "bgc": false,
      "nfeed": false,
      "avlnflg": false,
      "baseline": false,
      "dsb": false,
      "dsl": false,
      "dyn_lai": false
    },
    "eq": {
      "env": true,
      "bgc": true,
      "nfeed": true,
      "avlnflg": true,
      "baseline": true,
      "dsb": false,
      "dsl": true,
      "dyn_lai": true
    },
    "sp": {
      "env": true,
      "bgc": true,
      "nfeed": true,
      "avlnflg": true,
      "baseline": true,
      "dsb": false,
      "dsl": true,
      "dyn_lai": true
    },
    "tr": {
      "env": true,
      "bgc": true,
      "nfeed": true,
      "avlnflg": true,
      "baseline": true,
      "dsb": false,
      "dsl": true,
      "dyn_lai": true
    },
    "sc": {
      "env": true,
      "bgc": true,
      "nfeed": true,
      "avlnflg": true,
      "baseline": true,
      "dsb": false,
      "dsl": true,
      "dyn_lai": true
    }
  },
  "model_settings": {
    "dynamic_lai": 1
  }
}