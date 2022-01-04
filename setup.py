import pathlib
import distutils.core


distutils.core.setup(
    name='Renewable Test PSMs',
    version='0.1.0',
    author='Adriaan Hilbers',
    author_email='a.hilbers@icloud.com',
    url='https://github.com/ahilbers/renewable_test_psms',
    packages=['psm'],
    install_requires=pathlib.Path('requirements.txt').read_text().strip().split('\n')
)