from glob import glob

from setuptools import setup

PACKAGE = 'blueraven_skybeta'

setup(
    name=PACKAGE,
    version='0.1.0',
    packages=[PACKAGE],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + PACKAGE]),
        ('share/' + PACKAGE, ['package.xml']),
        ('share/' + PACKAGE + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Chelsea Yan',
    description="Team Blue Raven's mission node for the skybeta stack",
    license='MIT',
    entry_points={'console_scripts': ['mission = blueraven_skybeta.mission_node:main',
                                      'refnav_mission = blueraven_skybeta.refnav_node:main',
                                      'mppi_shadow = blueraven_skybeta.mppi_shadow:main',
                                      'yaw_offset_probe = blueraven_skybeta.yaw_offset_probe:main']},
)
