#!/usr/bin/env python3

from dataclasses import dataclass
from typing import List, Dict
from pathlib import Path

@dataclass
class Parameter:
    """Represents a single parameter in the CMT file"""
    name: str
    values: List[float]
    is_pft: bool  # True if parameter has PFT values, False if it's a global parameter

class CMTParser:
    """Parser for cmt_calparbgc.txt files"""
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.parameters: Dict[str, Parameter] = {}
        self._parse_file()
    
    def _parse_file(self):
        """Parse the CMT file and store parameters"""
        with open(self.file_path, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and header lines
            if not line or line.startswith('//=') or '//   PFT' in line:
                continue
                
            # Try to parse parameter line
            try:
                # Split on comment marker
                if '//' not in line:
                    continue
                    
                values_part, comment_part = line.split('//')
                
                # Extract parameter name from comment
                if ':' not in comment_part:
                    continue
                    
                param_name = comment_part.split(':')[0].strip()
                
                # Parse values
                values = [float(x) for x in values_part.split()]
                if not values:
                    continue
                
                # Determine if this is a PFT parameter
                is_pft = len(values) > 1
                
                self.parameters[param_name] = Parameter(
                    name=param_name,
                    values=values,
                    is_pft=is_pft
                )
            except (ValueError, IndexError):
                continue
    
    def get_parameter(self, name: str) -> Parameter:
        """Get a parameter by name"""
        return self.parameters.get(name)
    
    def get_all_parameters(self) -> Dict[str, Parameter]:
        """Get all parameters"""
        return self.parameters
    
    def get_pft_parameters(self) -> Dict[str, Parameter]:
        """Get only PFT parameters"""
        return {name: param for name, param in self.parameters.items() if param.is_pft}
    
    def get_global_parameters(self) -> Dict[str, Parameter]:
        """Get only global parameters"""
        return {name: param for name, param in self.parameters.items() if not param.is_pft}
    
    def to_lines(self, parameters: Dict[str, Parameter] = None, cmt_number: int = None, site_desc: str = "") -> List[str]:
        """
        Convert parameters back to file format
        
        Args:
            parameters: Optional dictionary of parameters to use instead of stored ones
            cmt_number: CMT number to include in the header
            site_desc: Site description to include in the header
        
        Returns:
            List of strings in the original file format
        """
        if parameters is None:
            parameters = self.parameters
        
        lines = []
        lines.append("//==========================================================")
        
        # Add CMT header if provided
        if cmt_number is not None:
            lines.append(f"// CMT{cmt_number} {site_desc}")
        
        # Add PFT header
        lines.append("//   PFT0         PFT1         PFT2         PFT3         PFT4         "
                    "PFT5         PFT6         PFT7         PFT8         PFT9")
        
        # First add PFT parameters
        pft_params = {k: v for k, v in parameters.items() if v.is_pft}
        for name, param in pft_params.items():
            values_str = "  " + "".join(f"{v:12.6f}" for v in param.values)
            lines.append(values_str + f" // {name}: ")
        
        # Then add global parameters
        global_params = {k: v for k, v in parameters.items() if not v.is_pft}
        for name, param in global_params.items():
            lines.append(f"{param.values[0]:10.6f}     // {name}: ")
        
        return lines

# def main():
#     """Example usage"""
#     import sys
    
#     if len(sys.argv) != 2:
#         print("Usage: python cmt_parser.py PATH_TO_CMT_FILE")
#         sys.exit(1)
    
#     parser = CMTParser(sys.argv[1])
    
#     print("PFT Parameters:")
#     for name, param in parser.get_pft_parameters().items():
#         print(f"{name}: {param.values}")
    
#     print("\nGlobal Parameters:")
#     for name, param in parser.get_global_parameters().items():
#         print(f"{name}: {param.values}")

# if __name__ == "__main__":
#     main() 