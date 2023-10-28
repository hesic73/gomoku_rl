from setuptools import setup, find_packages

setup(
    name="gomoku_rl",
    version="0.0.0",
    author="Sicheng He",
    author_email="hesicheng20@gmail.com",
    keywords=["RL", "pytorch"],
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchrl",
        "omegaconf",
        "hydra-core",
        "PyQt5",
    ],  # and other dependencies
)
