import os
from setuptools import setup
from glob import glob
package_name = 'pin_recognition'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yoshipi',
    maintainer_email='example@example.com',
    description='Pin tracking node',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points = {
        'console_scripts': [
            'pin_tracker_node = pin_recognition.pin_tracker_node:main',
        ],
    }

)
