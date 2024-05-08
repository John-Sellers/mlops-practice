# Importing necessary functions from setuptools module
from setuptools import find_packages, setup
from typing import List

hyphen_e_dot = '-e .'

def get_requirements(file_path: str) -> List[str]:
    """
    This function will return a list of requirements packages
    """

    requirements = []
    
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]

        if hyphen_e_dot in requirements:
            requirements.remove(hyphen_e_dot)
    
    return requirements

# Getting the list of requirements from requirements.txt
required_packages = get_requirements('requirements.txt')

# Setting up the package details
setup(
    # Name of the package
    name='mlproject',
    
    # Version number of the package
    version='0.0.1',
    
    # Author of the package
    author='John Sellers',
    
    # Author's email address
    author_email='sellers.e.john@gmail.com',
    
    # Finding all packages within the current directory
    packages=find_packages(),
    
    # List of required packages for installation
    install_requires=required_packages
)

print(required_packages)