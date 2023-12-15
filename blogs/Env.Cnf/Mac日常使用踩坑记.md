---
title: Mac日常使用踩坑记
date: 2022/10/19 13:48:35
categories:
    - 环境配置
tags: 
    - MacOS
---

用用MacOS😋

<!-- more -->

## 2023.12.15

homebrew 降级软件包

```sh
$brew unlink node
$brew link --overwrite --force node@20
```

## 2023.3.1

1. 安装 homebrew

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Homebrew是一款自由及开放源代码的软件包管理系统，用以简化macOS系统上的软件安装过程。

如果网络不好，可以使用镜像源

```bash
"$(curl -fsSL https://gitee.com/ineo6/homebrew-install/raw/master/install.sh)"
```



2. 安装 oh-my-zsh 并开启自动补全

```bash
# curl
sh -c "$(curl -fsSL https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh)"

# wegt 
sh -c "$(wget https://raw.githubusercontent.com/robbyrussell/oh-my-zsh/master/tools/install.sh -O -)"

# 国内镜像加速
wget https://gitee.com/mirrors/oh-my-zsh/raw/master/tools/install.sh && sudo chmod a+x install.sh && ./install.sh
```



```bash
git clone https://github.com/zsh-users/zsh-autosuggestions $ZSH_CUSTOM/plugins/zsh-autosuggestions
vim ~/.zshrc
# 加入插件列表
plugins=(git zsh-autosuggestions)
source ~/.zshrc
```



## 2022.11.30

1. 在vscode实现多文件编译

```bash
brew install ninja cmake tree
```

创建项目文件 Project，首先撰写 $CMakeLists.txt$文件，之后 `command+shift+p`打开cmake配置，选择 clang，自动开始构建 build 文件

CMakeLists.txt个人常用配置

```txt
cmake_minimum_required(VERSION 3.0) # 最低版本要求
project(XXX) # XXX为项目名称
include_directories(include/)
aux_source_directory(src/ DIR_SRCS)
set(EXECUTABLE_OUTPUT_PATH /Users/moguw/Desktop/Code/B-Designed/bin)
add_executable(main ${DIR_SRCS})
```

个人文件树如下所示

![](https://markdown-1308430375.cos.ap-nanjing.myqcloud.com/20221130155429.png)

2. mac下文件中文读写乱码的处理（原因：$locale$ 没有设置成 $utf-8$ ）

```bash
vim ~/.zshrc
export LC_ALL=zh_CN.UTF-8
export LANG=zh_CN.UTF-8
source ~/.zshrc
```

3. mac下写C++时，使用C++11标准的语法时可能会产生提醒

```bash
echo 'alias g++="g++ -std=c++11"' >> ./~zshrc
source ./~zshrc
```



## 2022.10.19

1. 破解软件绕过签名的办法
```bash
sudo xattr -rd com.apple.quarantine /Applications/xxxxxx.app
```
2. mac终端在粘贴时有多余字符：00~  ~01之类的
```bash
printf '\e[?2004l'
```
3. Mac 终端滚轮不滚页面，而是滚历史命令
```bash
tput rmcup
```
4. mac安装homebrew
```bash
/bin/zsh -c "$(curl -fsSL https://gitee.com/cunkai/HomebrewCN/raw/master/Homebrew.sh)" speed
```
5. mac卸载homebrew
```bash
/bin/zsh -c "$(curl -fsSL https://gitee.com/cunkai/HomebrewCN/raw/master/HomebrewUninstall.sh)"
```
