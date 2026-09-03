from setuptools import find_packages, setup

package_name = 'own_ship_gnc'

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
    maintainer='lars',
    maintainer_email='lars@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'os_transceiver_node = own_ship_gnc.os_transceiver_node:main',
            'live_plotter = own_ship_gnc.live_plotter:main',
        ],
    },
)
