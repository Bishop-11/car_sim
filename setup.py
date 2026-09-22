import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'car_sim'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='bishop',
    maintainer_email='bishop.prakash01@gmail.com',
    description='Lightweight 2D kinematic car + procedural road simulator with a simulated front camera, for learning classical and RL-based lane-following control.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sim_node = car_sim.sim_node:main',
            'manual_control_node = car_sim.manual_control_node:main',
        ],
    },
)
