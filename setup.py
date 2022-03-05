from setuptools import setup

with open("requirements.txt") as f:
    install_requires = f.read()

setup(
    name="gettint",
    version="0.1.1",
    install_requires=install_requires,
    url="https://github.com/OpusGang/gettint",
    author="OpusGang",
    packages=["gettint"],
    entry_points={
        "console_scripts": [
            "gettint = gettint:main"
            ]
        }
    )
