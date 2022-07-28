import setuptools

setuptools.setup(
    name="shut-the-box",
    version="0.0.1",
    author="",
    author_email="georges.lebellier@sony.com",
    description="Reinforcement learning applied to Shut the box 9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "."},
    packages=setuptools.find_packages(where="."),
    python_requires=">=3.6",
)