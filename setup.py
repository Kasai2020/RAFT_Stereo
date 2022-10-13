import subprocess
import sys
from setuptools import Extension
from setuptools import setup
import os

setup(
    name='RAFT_Stereo',
    version='1.0.0',
    install_requires=['numpy'],
    license='MIT',
    maintainer='Isaac Kasahara',
    maintainer_email='i.kasahara@partner.samsung.com',
    url='https://gitlab.saicny.com/saic-ny/perception/glass_detection',
)