from setuptools import setup


setup(
    name='FastformerLM',
    version='0.1.0',
    author='Michael Hu',
    author_email='prmhu@yahoo.com',
    url='https://github.com/mtanghu/Additive-Attention-Is-Not-All-You-Need-Maybe',
    description=(
        'Fastformer/Additive Attention for causal language modeling'
    ),
    py_modules=['fastformer'],
    install_requires=[
        'torch>=1.0.0',
        'transformers[torch]',
        'datasets',
        'pandas',
        'numpy',
        'matplotlib'
    ]
)