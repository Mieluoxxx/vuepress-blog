---
title: Archlinux安装Clash
date: 2023/10/21 08:34:32
categories:
    - 环境配置
tags: 
    - Archlinux
---

## 安装
Github release 链接：[https://github.com/Fndroid/clash_for_windows_pkg/releases](https://github.com/Fndroid/clash_for_windows_pkg/releases)

在其中找到需要的版本，然后下载即可，如果国内下载较慢的话，可以使用大佬提供的代理1，使用方式是在下载链接的前面加上 `https://ghproxy.com/` 即可，则代理后的链接为
`https://ghproxy.com/https://github.com/Fndroid/clash_for_windows_pkg/releases/download/0.20.34/Clash.for.Windows-0.20.34-x64-linux.tar.gz`

下载后应该是名为 Clash.for.Windows-version-x64-linux.tar.gz 的文件，然后解压：
`tar -zxvf Clash.for.Windows.xxx` 
解压后会出现对应的文件夹，文件夹中会有一个名为 cfw 的文件，按理说现在在文件中直接打开终端，然后运行
`./cfw` 
就会直接打开 Clash For Windows 了，与 Windows 版本基本平常无差。



## 方便使用

接下来最好可以链接一个启动项来方便配置，不过，是可选的：

`mkdir -p ~/.local/bin`

`ln -s /home/user/Downloads/xxxx/cfw /home/user/.local/bin/cfw` # 软链接

`export PATH=/home/moguw/.local/bin:$PATH` # 环境变量



```bash
sudo vim ~/.local/share/applications/clash.desktop

# 写入如下内容

[Desktop Entry]
Name=Clash Fow Windows
Exec=/home/user/.local/bin/cfw
Icon=/home/user/.local/bin/cfw
Type=Application
StartupNotify=true
```



`sudo chmod +x ~/.local/share/applications/clash.desktop`



