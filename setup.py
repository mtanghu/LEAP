from setuptools import setup, find_packages


setup(
    name='leap-transformer',
    version='0.1.1',
    license='CC0 1.0 Universal',
    author='Michael Hu',
    author_email='prmhu@yahoo.com',
    url='https://github.com/mtanghu/Additive-Attention-Is-Not-All-You-Need-Maybe',
    description=(
        'Linear Explainable Attention in Parallel (LEAP) for causal language modeling (also implements fastformer)'
    ),
    packages=find_packages('src'),
    package_dir={'': 'src'},
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords='linear transformer NLP deep learning pytorch',
    install_requires=[
        'torch>=1.0.0',
        'transformers[torch]',
        'datasets',
        'pandas',
        'numpy',
        'matplotlib'
    ]
)