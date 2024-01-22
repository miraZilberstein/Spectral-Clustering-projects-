from distutils.core import setup, Extension

module = Extension("mykmeanssp", sources = ["kmeans.c"])

setup(name="K-Means", version="1.0", description="This is a package for mykmeanssp", ext_modules=[module])
