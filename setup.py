from setuptools import setup, find_packages, Extension
#from distutils.core import setup, Extension




VBB = Extension('_VBBinaryLensingLibrary',
		 sources=['./pyLIMA/subroutines/VBBinaryLensingLibrary/VBBinaryLensingLibrary.cpp',
			  './pyLIMA/subroutines/VBBinaryLensingLibrary/VBBinaryLensingLibrary.i',

		 ],
		swig_opts=['-c++','-modern', '-I../include'],

		)

setup(
    name="pyLIMA",
    version="0.1.5",
    description="Microlsening analysis package.",
    keywords='Microlsening analysis package.',
    author="Etienne Bachelet",
    author_email="etibachelet@gmail.com",
    license='GPL-3.0',
    url="http://github.com/ebachelet/pyLIMA",
    download_url = 'https://github.com/ebachelet/pyLIMA/archive/0.1.tar.gz',
    packages=find_packages('.'),
    include_package_data=True,
    install_requires=['scipy','numpy','matplotlib','astropy','emcee','pyslalib'],
    python_requires='>=2.7,<4',
    test_suite="nose.collector",
    ext_modules = [VBB],
    py_modules = ["VBBinaryLensingLibrary"],
    classifiers=[
		'Development Status :: 5 - Production/Stable',
		'Intended Audience :: Developers',
		'Topic :: Software Development :: Build Tools',
                'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
		'Programming Language :: Python :: 2.7',
		'Programming Language :: Python :: 3',	   
],
    package_data={
    'sample': ['Claret2011.fits','Yoo_B0B1.dat'],
},

       
)
