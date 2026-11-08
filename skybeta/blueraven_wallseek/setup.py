from glob import glob

from setuptools import setup

PACKAGE = 'blueraven_wallseek'

setup(
    name=PACKAGE,
    version='0.1.0',
    packages=[PACKAGE],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + PACKAGE]),
        ('share/' + PACKAGE, ['package.xml']),
        ('share/' + PACKAGE + '/launch', glob('launch/*.launch.py')),
        ('share/' + PACKAGE + '/route', glob('route/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Darion',
    description="Team Blue Raven: wall-seek mission (forward to a wall, translate along it, land past its end)",
    license='MIT',
    entry_points={'console_scripts': ['wallseek = blueraven_wallseek.wallseek_node:main']},
)
