from setuptools import find_packages, setup

package_name = 'pin_recognition'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='cielo',
    maintainer_email='cielo@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'pin_tracker_node = pin_recognition.pin_tracker_node:main',
            'fake_pin_camera_node = pin_recognition.fake_pin_camera_node:main',
            'easytracker_node = pin_recognition.easytracker_node:main',
        ],
    },

)
