from setuptools import setup, find_packages



# put a disclaimer in the readme for publishing on pypi
description = open("README.md", "r", encoding="utf-8").read().split("\n")
description.insert(2, "NOTE: The description shown here is just the github README. As such, some of the equations may not render. Checkout out the github for a more refined description where you can also see the code and contribute!")
description.insert(3, '')



setup(
    name='leap-transformer',
    version='0.1.8',
    license='CC0 1.0 Universal',
    author='Michael Hu',
    author_email='prmhu@yahoo.com',
    url='https://github.com/mtanghu/Additive-Attention-Is-Not-All-You-Need-Maybe',
    description=(
        'Linear Explainable Attention in Parallel (LEAP) for causal language modeling (also implements fastformer)'
    ),
    packages=find_packages('src'),
    package_dir={'': 'src'},
    long_description='\n'.join(description),
    long_description_content_type='text/markdown',
    keywords='linear transformer NLP deep learning pytorch',
    install_requires=[
        'transformers',
        'datasets',
        'pandas',
        'numpy',
        'matplotlib'
    ]
)