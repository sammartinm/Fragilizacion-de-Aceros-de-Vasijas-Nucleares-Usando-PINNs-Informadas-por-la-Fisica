from setuptools import find_packages, setup

setup(
    name='src',                # El nombre de tu paquete
    packages=find_packages(),  # Busca automáticamente la carpeta con __init__.py
    version='0.1.0',
    description='Código del TFM sobre PINNs',
    author='Tu Nombre',
)