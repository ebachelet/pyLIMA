from setuptools import setup, find_packages
from distutils.core import setup, Extension




VBB = Extension('_VBBinaryLensingLibrary',
		 sources=['./pyLIMA/subroutines/VBBinaryLensingLibrary/VBBinaryLensingLibrary.cpp',
			  './pyLIMA/subroutines/VBBinaryLensingLibrary/VBBinaryLensingLibrary.i',
		 ],
		swig_opts=['-c++','-modern', '-I../include'],

		)

setup(
    name="pyLIMA",
    version="0.1",
    description="Microlsening analysis package.",
    author="Etienne Bachelet",
    author_email="ebachelet@lcogt.net",
    url="http://github.com/ebachelet/pyLIMA",
    packages=find_packages('.'),
    include_package_data=True,
    test_suite="nose.collector",
    ext_modules = [VBB],
    py_modules = ["VBBinaryLensingLibrary"],
       
)
