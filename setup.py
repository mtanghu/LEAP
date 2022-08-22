from setuptools import setup, find_packages


setup(
    name='fastformerLM',
    version='0.1.2',
    license='CC0 1.0 Universal',
    author='Michael Hu',
    author_email='prmhu@yahoo.com',
    url='https://github.com/mtanghu/Additive-Attention-Is-Not-All-You-Need-Maybe',
    description=(
        'Fastformer, a Linear Transformer using Additive Attention for causal language modeling'
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