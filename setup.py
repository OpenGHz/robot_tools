from setuptools import setup, find_packages

setup(
    name="robotics_tools",
    version="1.0.2",
    packages=find_packages(),
    package_dir={"": "."},
    install_requires=[
        "numpy>=1.19.5",
        "scipy",
        "matplotlib",
        "python-socketio",
        "eventlet",
        "python-xlib",
    ],
    author="OpenGHz",
    author_email="ghz23@mails.tsinghua.edu.cn",
    description="Robot Tools for develop your robots.",
    url="https://github.com/OpenGHz/robot_tools.git",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
