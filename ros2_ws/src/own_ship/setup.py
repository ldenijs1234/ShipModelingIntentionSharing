import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'own_ship'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lars',
    maintainer_email='lars@todo.todo',
    description='Own Ship GNC & Collision Avoidance package',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'own_ship_node = own_ship.own_ship_node:main',
            'live_plotter_node = own_ship.live_plotter_node:main',
        ],
    },
)