import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setuptools.setup(
    name='yolow',
    version='0.1',
    description='YOLO-World (w/o third-party)',
    long_description='',
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(exclude=['.vscode', '.idea']),
    python_requires='>=3.6',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GPL3.0 License',
        'Operating System :: OS Independent',
    ],
    install_requires=required,
)
