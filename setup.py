from setuptools import setup, find_packages

setup(
    name='disease_detection',
    version='0.2.1',
    packages=find_packages(),
    package_data={'disease_detection': ['datasets/cancer.csv', 'datasets/diabetes.csv']},
    install_requires=[
        'numpy',
        'matplotlib',
        'pandas',
        'scikit-learn',
        'seaborn',
        'pytest'
    ],
    
    author='Niraj Prasad',
    author_email='nirajsaiprasad@gmail.com',
    description='A package for disease detection',
    #long_description=open('README.md').read(),
    #long_description_content_type='text/markdown',
    url='https://github.com/nirajprasad12/ds5010-disease-detection',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
