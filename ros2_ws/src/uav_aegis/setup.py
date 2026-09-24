from setuptools import setup
import os
from glob import glob

package_name = 'uav_aegis'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'scripts'), glob('scripts/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Rhutvik Pachghare',
    maintainer_email='rhutvik@example.com',
    description='UAV Fault Diagnostics with Deep Learning',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'fault_inference_node = uav_aegis.fault_inference_node:main',
            'px4_log_replay = uav_aegis.px4_log_replay:main',
        ],
    },
)