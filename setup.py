from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT='-e .'
def get_requirements(file_path:str)->List[str]:
    '''This function will return the list of requirements'''
    requirements=[]
    with open(file_path) as f:
        requirements=f.readlines()
        requirements=[req.replace("/n","") for req in requirements] #when we read lines the next line will contain /n inorder to remove it we use this function

        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)

    return requirements
setup(
    name='mlproject',
    version='0.0.1',
    author='Siva',
    author_email='sivacharany2001@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)