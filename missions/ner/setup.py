#nsml: nsml/default_ml:tf-1.4.1

from distutils.core import setup

setup(
    name='NER_NSML_Baseline',
    version='1',
    description='NER_NSML_Baseline',
    install_requires=[
        'tensorflow-gpu==1.4.1',
        'numpy>=1.11.0'
    ]
)
