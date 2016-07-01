from setuptools import setup, find_packages

setup(
    name="pyLIMA",
    version="0.1",
    description="Microlsening analysis package.",
    author="Etienne Bachelet",
    author_email="ebachelet@lcogt.net",
    url="http://github.com/ebachelet/pyLIMA",
    packages=find_packages('.'),
    test_suite="nose.collector",
)
