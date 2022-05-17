##### How to upload to PyPi:
# python setup.py sdist
# twine upload dist/*

from setuptools import setup
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()


setup(
    name='WassersteinTSNE',
    version='1.1.1',    
    description='A package for dimensionality reduction of probability distributions',
    url='https://github.com/fsvbach/WassersteinTSNE',
    author='Fynn Bachmann, Philipp Hennig, Dmitry Kobak',
    author_email='fynn.bachmann@uni-hamburg.de',
    license='MIT',
    packages=['WassersteinTSNE'],
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=['numpy',
                      'scipy',
                      'pandas',
                      'openTSNE',
                      'scikit-learn',
                      'igraph',
                      'leidenalg',
                      'matplotlib',                    
                      ],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
