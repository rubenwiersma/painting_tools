import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='painting_tools',
    version='0.0.1',
    author='Ruben Wiersma',
    author_email="rubenwiersma@gmail.com",
    description='Tools to study paintings using (multi-)spectral data, for example for pigment identification.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rubenwiersma/painting_tools",
    project_urls={
        "Bug Tracker": "https://github.com/rubenwiersma/painting_tools/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(where=".", include=["painting_tools", "painting_tools.*"]),
    python_requires='>=3.9',
    install_requires=[
        'jax[cuda]',
        'jaxopt',
        'cvxopt',
        'dm-haiku',
        'matplotlib',
        'spectral',
        'kornia',
        'jupyter'
    ],
)
