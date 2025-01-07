from setuptools import setup, find_packages

# Lê as dependências do arquivo requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="app",
    version='0.0.1',
    description="Bot para apoio de diagnostico",
    author="Guilherme Fernando Angelico",
    author_email="angelicogfa@gmail.com",
    packages=find_packages(
        include=[
            "app",
            "app.*",
        ]
    ),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.8",
)