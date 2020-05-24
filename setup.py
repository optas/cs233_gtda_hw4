from setuptools import setup

setup(name='cs233_gtda_hw4',
      version='0.2',
      description='Fourth assignment for CS-233, Stanford',
      url='http://github.com/optas/cs233_gtda_hw4',
      author='Panos Achlioptas',
      author_email='pachlioptas@gmail.com',
      license='MIT',
      install_requires=['torch',
                        'numpy',
                        'scikit-learn',
                        'matplotlib',
                        'tqdm'],
      packages=['cs233_gtda_hw4'],
      zip_safe=False)
