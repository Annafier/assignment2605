from setuptools import setup, find_packages

setup(
    name='atrumod',
    version='0.1.0',
    description='ATR-UMOD Multimodal Oriented Object Detection',
    packages=find_packages(),
    python_requires='>=3.10',
    install_requires=[
        'torch>=2.0.0',
        'mmcv>=2.0.0',
        'mmdet>=3.0.0',
        'mmengine>=0.7.0',
        'mmrotate>=1.0.0',
        'numpy',
        'opencv-python',
        'einops',
        'tensorboard',
    ],
)
