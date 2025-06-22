from setuptools import setup, find_packages

setup(
    name="multiphysics_fdm",  # 库的名称，安装和导入时用
    version="1.0",  # 版本号
    packages=find_packages(),  # 自动发现所有包含 __init__.py 的包
    author="XuXiaoNan",  # 你的名字
    author_email="1936252019@qq.com",  # 你的邮箱
    description="A custom library for multiphysics simulation with FDM",  # 库的描述
    #long_description=open("README.md").read() if exists("README.md") else "",  # 如果有 README.md 可填
    long_description_content_type="text/markdown",
    url="https://github.com/DQ212XXN/multiphysics_fdm",  # 仓库地址，没有可填自己代码存放地方
    classifiers=[  # 分类信息，方便索引，可选
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.11',  # Python 版本要求
)