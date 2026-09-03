from setuptools import find_packages, setup

package_name = 'gnc_network_layer'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lars',
    maintainer_email='lars@todo.todo',
    description='Network simulation layer and latency injection bridge',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'latency_bridge_node = gnc_network_layer.latency_bridge_node:main',
            'ts_simulator_node = gnc_network_layer.ts_simulator_node:main',
        ],
    },
)
