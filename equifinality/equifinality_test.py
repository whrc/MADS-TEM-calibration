#!/usr/bin/env python3

import os
import sys
import json
import shutil
import numpy as np
from pathlib import Path
from cmt_parser import CMTParser, Parameter

class EquifinalityTest:
    def __init__(self, cmt_number, cmt_source_dir, dvmdostem_dir, input_data_dir):
        """
        Initialize the equifinality test for a specific CMT
        
        Args:
            cmt_number: The CMT number to test (e.g., 50)
            cmt_source_dir: Directory containing the source cmt_calparbgc.txt and other config files
            dvmdostem_dir: Path to the dvmdostem directory containing the executable
            input_data_dir: Directory containing input data files (climate, vegetation, etc.)
        """
        self.cmt_number = cmt_number
        self.cmt_source_dir = Path(cmt_source_dir)
        self.dvmdostem_dir = Path(dvmdostem_dir)
        self.input_data_dir = Path(input_data_dir)
        self.dvmdostem_config = self.read_base_config()
        
        # Create output directory with CMT-specific name
        self.output_dir = Path(f"cmt{cmt_number}_equifinality_tests")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Parameters to control variations
        self.variation_config = {
            'variation_range': 0.05,  # ±5% variation by default
            'fixed_parameters': [],   # List of parameter names to keep fixed
            'tem_args': {
                'p': 100,  # Example values
                'e': 100,
                's': 100,
                't': 100,
                'n': 100
            }
        }
    
    def read_base_config(self):
        """Read the base config.js file"""
        config_path = self.dvmdostem_dir / "config" / "config.js"
        with open(config_path, 'r') as f:
            # Remove comments before parsing JSON
            config_lines = []
            for line in f:
                if '//' in line:
                    line = line.split('//')[0]
                config_lines.append(line)
            config_text = ''.join(config_lines)
            
            try:
                return json.loads(config_text)
            except json.JSONDecodeError as e:
                print(f"Error parsing config.js: {str(e)}")
                sys.exit(1)
    
    def create_test_config(self, test_dir, params_dir, config_dir, output_dir):
        """Create a modified config.js for this test variation"""
        config = self.dvmdostem_config.copy()
        
        # Update paths in the config with absolute paths
        config['IO']['parameter_dir'] = str(params_dir.resolve()) + "/"
        config['IO']['output_dir'] = str(output_dir.resolve()) + "/"
        config['IO']['output_spec_file'] = str((config_dir / "output_spec.csv").resolve())
        
        # Update input data paths
        input_paths = {
            'hist_climate_file': 'historic-climate.nc',
            'proj_climate_file': 'projected-climate.nc',
            'veg_class_file': 'vegetation.nc',
            'drainage_file': 'drainage.nc',
            'soil_texture_file': 'soil-texture.nc',
            'co2_file': 'co2.nc',
            'proj_co2_file': 'projected-co2.nc',
            'runmask_file': 'run-mask.nc',
            'topo_file': 'topo.nc',
            'fri_fire_file': 'fri-fire.nc',
            'hist_exp_fire_file': 'historic-explicit-fire.nc',
            'proj_exp_fire_file': 'projected-explicit-fire.nc'
        }
        
        # Update each input path in the config
        for key, filename in input_paths.items():
            file_path = self.input_data_dir / filename
            if file_path.exists():
                config['IO'][key] = str(file_path.resolve())
            else:
                print(f"Warning: Input file {filename} not found in {self.input_data_dir}")
        
        # Write the modified config to the config directory
        config_path = config_dir / "config.js"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return config_path
        
    def set_variation_range(self, range_percent):
        """Set the variation range as a percentage"""
        self.variation_config['variation_range'] = range_percent / 100.0
        
    def set_fixed_parameters(self, params):
        """Set which parameters should not be varied"""
        self.variation_config['fixed_parameters'] = params
        
    def set_tem_args(self, p=None, e=None, s=None, t=None, n=None):
        """Set TEM executable arguments"""
        if p is not None: self.variation_config['tem_args']['p'] = p
        if e is not None: self.variation_config['tem_args']['e'] = e
        if s is not None: self.variation_config['tem_args']['s'] = s
        if t is not None: self.variation_config['tem_args']['t'] = t
        if n is not None: self.variation_config['tem_args']['n'] = n
        
    def setup_test_directory(self, variation_id):
        """Create a test directory with all necessary configuration files"""
        # Create variation-specific directory
        test_dir = self.output_dir / str(variation_id)
        test_dir.mkdir(parents=True, exist_ok=True)
        
        # Create parameters, config, and output subdirectories
        params_dir = test_dir / "parameters"
        config_dir = test_dir / "config"
        output_dir = test_dir / "output"
        
        for directory in [params_dir, config_dir, output_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Copy all configuration files from the source CMT directory to parameters subdirectory
        for file in self.cmt_source_dir.glob("cmt_*.txt"):
            if file.name != "cmt_calparbgc.txt":
                shutil.copy2(file, params_dir)
        
        # Copy output_spec.csv to config directory
        output_spec_src = self.dvmdostem_dir / "config" / "output_spec.csv"
        if output_spec_src.exists():
            shutil.copy2(output_spec_src, config_dir)
        else:
            print(f"Warning: {output_spec_src} not found")
        
        return test_dir, params_dir, config_dir, output_dir
    
    def should_vary_parameter(self, param_name):
        """Check if a parameter should be varied"""
        return param_name not in self.variation_config['fixed_parameters']
    
    def generate_parameter_variation(self, base_params):
        """Generate a variation of the parameters for testing"""
        # Parse the base parameters file
        parser = CMTParser(base_params)
        parameters = parser.get_all_parameters()
        
        # Create variations of parameters
        varied_parameters = {}
        range_val = self.variation_config['variation_range']
        
        for name, param in parameters.items():
            if self.should_vary_parameter(name):
                # Generate variations for each value
                varied_values = [
                    v * (1 + np.random.uniform(-range_val, range_val))
                    for v in param.values
                ]
                varied_parameters[name] = Parameter(
                    name=name,
                    values=varied_values,
                    is_pft=param.is_pft
                )
            else:
                # Keep original values for fixed parameters
                varied_parameters[name] = param
        
        # Get site description from source directory name
        site_desc = self.cmt_source_dir.name.replace(f"CMT{self.cmt_number}-", "").replace("-", " ")
        
        # Convert back to file format with CMT header
        return parser.to_lines(varied_parameters, cmt_number=self.cmt_number, site_desc=site_desc)
    
    def run_single_test(self, variation_id):
        """Run a single test with a parameter variation"""
        # Setup test directory
        test_dir, params_dir, config_dir, output_dir = self.setup_test_directory(variation_id)
        
        # Generate parameter variation from the source CMT directory
        base_params = self.cmt_source_dir / "cmt_calparbgc.txt"
        varied_params = self.generate_parameter_variation(base_params)
        
        # Write varied parameters to the parameters subdirectory
        with open(params_dir / "cmt_calparbgc.txt", 'w') as f:
            f.write('\n'.join(varied_params))
        
        # Create modified config.js for this test
        config_path = self.create_test_config(test_dir, params_dir, config_dir, output_dir)
        
        # Construct dvmdostem command with absolute paths
        args = self.variation_config['tem_args']
        dvmdostem_exe = self.dvmdostem_dir / "dvmdostem"
        cmd = (f"{dvmdostem_exe.resolve()} -f {config_path.resolve()} "
               f"-p {args['p']} -e {args['e']} -s {args['s']} "
               f"-t {args['t']} -n {args['n']}")
        
        print(f"Test {variation_id} setup complete in {test_dir}")
        print(f"Command: {cmd}")
        return {
            'variation_id': variation_id,
            'test_dir': str(test_dir.resolve()),
            'params_dir': str(params_dir.resolve()),
            'config_path': str(config_path.resolve()),
            'output_dir': str(output_dir.resolve()),
            'command': cmd
        }
    
    def run_tests(self, n_variations=10):
        """Run multiple tests"""
        results = []
        for i in range(n_variations):
            try:
                result = self.run_single_test(i)
                results.append(result)
            except Exception as e:
                print(f"Error in test {i}: {str(e)}")
        return results

def main():
    if len(sys.argv) != 5:
        print("Usage: python equifinality_test.py CMT_NUMBER CMT_SOURCE_DIR DVMDOSTEM_DIR INPUT_DATA_DIR")
        sys.exit(1)
    
    cmt_number = int(sys.argv[1])
    cmt_source_dir = sys.argv[2]
    dvmdostem_dir = sys.argv[3]
    input_data_dir = sys.argv[4]

    # Initialize tester
    tester = EquifinalityTest(cmt_number, cmt_source_dir, dvmdostem_dir, input_data_dir)

    tester.set_variation_range(5)  # 5% variation
    # todo: only tweak the passed parameters
    tester.set_fixed_parameters(['kra'])  # Keep kra parameter fixed
    tester.set_tem_args(p=100, e=200, s=100, t=50, n=75)

    # Run tests
    results = tester.run_tests(n_variations=3)

    # Print summary
    print("\nTest Summary:")
    for result in results:
        print(f"Variation {result['variation_id']}:")
        print(f"  Directory: {result['test_dir']}")
        print(f"  Config: {result['config_path']}")
        print(f"  Command: {result['command']}\n")


# todo: look at chatgpt response when generating the parameters
# todo: run your tests using toolik data in the repo, cmt05
# todo: sa-demo-config.yaml, dvm-dos-tem/mads_calibration
# todo: sa-setup-and-run.py
if __name__ == "__main__":
    main() 