import this
from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent


VERSION = '0.0.1'
DESCRIPTION = 'A basic gradio class and class function wrapper'
LONG_DESCRIPTION = (this_directory / "README.md").read_text()

# Setting up
setup(
    name="gradio_wrap",
    version=VERSION,
    author="LucaVivona (Luca Vivona)",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=['gradio'],
    keywords=['python', 'sockets', 'artificial intelligence', 'machine learning', 'visualizatio', 'wrapper'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)