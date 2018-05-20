import setuptools


setuptools.setup(
    author="Rany Keddo",
    extras_require={
        "test": [
            "pytest"
        ]
    },
    install_requires=[
        "numpy",
        "pandas",
        "tensorflow==1.8.0"
    ],
    license="MIT",
    name="char-cnn",
    description="Tensorflow implementation of Char-CNN: Character-level Convolutional Networks for Text Classification, Zhang et al, 2016",
    keywords="tensorflow character char cnn nlp deep-learning",
    packages=setuptools.find_packages(
        exclude=[
            "tests"
        ]
    ),
    url="https://github.com/reflectionlabs/char-cnn",
    version="0.1.2"
)
