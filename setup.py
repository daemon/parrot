import setuptools


setuptools.setup(
    name='parrot',
    version='0.0.1',
    author='Raphael Tang',
    find_packages=setuptools.find_packages(),
    install_requires=open('requirements.txt').read().splitlines(),
    entry_points={
        'console_scripts': [
            'parrot_eval = parrot.run.eval:main',
        ],
    },
    python_requires='>=3.10',
)