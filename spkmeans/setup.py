from distutils.core import setup, Extension

module = Extension("spkmeansmodule", sources = ["spkmeans.c", "spkmeansmodule.c"])

setup(name="K-Means", version="1.0", description="This is a package for spkmeansmodule", ext_modules=[module])
