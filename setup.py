from setuptools import setup, find_packages
from distutils.util import convert_path



meta = {}
with open(convert_path("lib/version.py")) as f:
    exec(f.read(), meta)

setup(
    name="humansensormotion",
    version=meta["__version__"],
    description="Python package for analyzing sensor-collected human motion "
    "data (e.g. physical activity levels, gait dynamics, bone health) based on the work of sensormotion sho-87" ,
    author="MC",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3",
    install_requires=["matplotlib", "numpy", "scipy", "pandas"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research/Engineering",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering",
    ],
    keywords="gait accelerometer signal-processing walking actigraph physical-activity",
    zip_safe=True,
)