from setuptools import setup, find_packages

setup(
    name='QICAS-QIO',
    version='1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'pyscf',
        'block2'
    ],
    author='Ke Liao & Lexin Ding',
    author_email='ke.liao.whu@gmail.com',
    description='Quantum information-based orbital optimization for quantum chemistry calculations.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/LexinDing/QICAS/tree/master',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)