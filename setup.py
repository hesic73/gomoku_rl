from setuptools import setup, find_packages

setup(
    name="gomoku_rl",
    version="0.0.0",
    author="hesic73",
    author_email="hesicheng20@gmail.com",
    keywords=["RL", "pytorch"],
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchrl==0.2.1",
        "omegaconf",
        "hydra-core",
        "tqdm",
        "wandb",
        "matplotlib",
    ],  # and other dependencies
)
