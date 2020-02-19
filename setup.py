from setuptools import setup, find_packages, Extension

setup(
    name="pyLIMA",
    version="0.8.2",
    description="Microlsening analysis package.",
    keywords='Microlsening analysis package.',
    author="Etienne Bachelet",
    author_email="etibachelet@gmail.com",
    license='GPL-3.0',
    url="http://github.com/ebachelet/pyLIMA",
    download_url = 'https://github.com/ebachelet/pyLIMA/archive/0.1.tar.gz',
    packages=find_packages('.'),
    include_package_data=True,
    install_requires=['scipy','numpy','matplotlib','astropy','emcee','numba','bokeh','PyAstronomy','VBBinaryLensing','Cython'],
    python_requires='>=3.6,<4',
    test_suite="nose.collector",
    classifiers=[
		'Development Status :: 5 - Production/Stable',
		'Intended Audience :: Developers',
		'Topic :: Software Development :: Build Tools',
                'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
		'Programming Language :: Python :: 3',	   
],
    package_data={
    '': ['Claret2011.fits','Yoo_B0B1.dat'],
},
     
)
