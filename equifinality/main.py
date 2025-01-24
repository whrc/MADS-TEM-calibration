import pandas as pd
import os

def read_excel_file(file_path='calibration_results.xlsx'):
    sheets = pd.read_excel(file_path, sheet_name=None)
    return sheets

def create_cmt_folder_name(sheet_name, site_name):
    """Create a standardized folder name from CMT and site name"""
    # Clean up site name: replace spaces with hyphens and remove special characters
    clean_site = site_name.strip().replace(' ', '-').replace('/', '-')
    # Remove any double hyphens and other special characters
    clean_site = '-'.join(filter(None, clean_site.split('-')))
    return f"{sheet_name}-{clean_site}"

def format_value(value):
    """Format a value to string with proper handling of NaN and non-numeric values"""
    try:
        if pd.isna(value):
            return f"{0.0:12.6f}"
        return f"{float(value):12.6f}"
    except (ValueError, TypeError):
        return f"{0.0:12.6f}"

def generate_calparbgc(sheet_name, df, output_dir=None):
    """
    Generate a calparbgc.txt file from a given sheet's data
    
    Args:
        sheet_name (str): Name of the sheet (e.g., 'CMT61')
        df (pd.DataFrame): DataFrame containing the calibration data
        output_dir (str, optional): Output directory path
    """
    try:
        # Get site name from the first row
        site_name = df['Sitre Name'].iloc[0] if 'Sitre Name' in df.columns else df[' '].iloc[0]
        
        # Create folder name and path
        folder_name = create_cmt_folder_name(sheet_name, site_name)
        if output_dir:
            folder_path = os.path.join(output_dir, folder_name)
        else:
            folder_path = folder_name
            
        # Create the folder if it doesn't exist
        os.makedirs(folder_path, exist_ok=True)
        
        # Set output file path
        output_file = os.path.join(folder_path, "cmt_calparbgc.txt")
        
        # Parameters that need PFT values (in order they should appear)
        pft_params = ['cmax', 'nmax', 
                      'cfall(0)', 'cfall(1)', 'cfall(2)',
                      'nfall(0)', 'nfall(1)', 'nfall(2)',
                      'kra', 'krb(0)', 'krb(1)', 'krb(2)', 'frg']
        
        # Global parameters (no PFT)
        global_params = ['micbnup', 'kdcrawc', 'kdcsoma', 'kdcsompr', 'kdcsomcr']
        
        # Start building the output
        lines = []
        lines.append("//==========================================================")
        lines.append(f"// {sheet_name} // {site_name}")
        lines.append("//   PFT0         PFT1         PFT2         PFT3         PFT4         PFT5         PFT6         PFT7         PFT8         PFT9")
        
        # Add PFT parameters
        for param in pft_params:
            values = []
            param_data = df[df['Parameters'] == param]
            
            # Get values for each PFT (0-9)
            for pft in range(10):
                pft_row = param_data[param_data['PFT'] == float(pft)]
                value = pft_row['Calibrated'].iloc[0] if not pft_row.empty else 0.0
                values.append(format_value(value))
            
            # Join values with proper spacing and add parameter name
            line = "  " + "".join(values) + f" // {param}: "
            lines.append(line)
        
        # Add global parameters
        for param in global_params:
            param_data = df[df['Parameters'] == param]
            if not param_data.empty:
                value = param_data['Calibrated'].iloc[0]
                lines.append(f"{format_value(value).strip()}     // {param}: ")
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write('\n'.join(lines))
        
        return output_file
    
    except Exception as e:
        print(f"Error processing {sheet_name}: {str(e)}")
        return None

if __name__ == "__main__":
    sheets = read_excel_file()
    
    # Create a base output directory for all CMT folders
    base_output_dir = "calibration_results"
    os.makedirs(base_output_dir, exist_ok=True)
    
    # Generate files for all CMT sheets
    for sheet_name, df in sheets.items():
        if sheet_name.startswith("CMT"):
            output_file = generate_calparbgc(sheet_name, df, base_output_dir)
            if output_file:
                print(f"Generated {output_file}")