from setuptools import setup, find_packages

setup(
    name='religion-prediction',
    version='0.0.1',
    # url='https://github.com/saadgulzar/india_names',
    author='Saad Gulzar, Tanushree Goyal, Feyaad Allie and Gufran Pathan',
    description='Description of my package',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    # install_requires=['numpy >= 1.11.1', 'matplotlib >= 1.5.1'],
    install_requires=[
        'scikit-learn==1.2.0','pandas'
    ],

)