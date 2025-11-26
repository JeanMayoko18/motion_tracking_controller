from setuptools import find_packages, setup

package_name = 'motion_tracking_controller'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/mujoco.launch.py',
            'launch/real.launch.py',
        ]),
    ],
    install_requires=[
        'setuptools',
    ],
    zip_safe=True,
    maintainer='Jean Chrysostome Mayoko Biong',

    maintainer_email='icmayoko18@gmail.com',
    description='Run RL policies (MuJoCo or real robot) from W&B or local ONNX.',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'policy_node = motion_tracking_controller.policy_node:main',
        ],
    },
)