from setuptools import find_packages, setup
from pathlib import Path

PARENT = Path(__file__).parent
README = (PARENT / 'README.md').read_text(encoding='utf-8')

setup(
    name='traffic-vision-ai',
    version='0.1.0',
    description='AI Smart Sensor for road traffic monitoring, detection, tracking and indicators',
    long_description=README,
    long_description_content_type='text/markdown',
    author='Hamza Elouiaazzani',
    url='https://github.com/hamzaelouiaazzani/traffic-sensor-ai',
    
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    install_requires=(PARENT / 'requirements.txt').read_text().splitlines(),
    python_requires='>=3.8',
)