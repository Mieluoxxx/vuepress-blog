---
title: ArchLinux(EndeavorOS)开局
date: 2023/10/21 08:34:32
categories:
    - 环境配置
tags: 
    - Archlinux
    - Clash
---

## EndeavorOS在线安装
```bash
nano /etc/calamares/modules/welcome_online.conf
# 将检查的网址替换为www.baidu.com
```


## 安装中文输入法
```bash
sudo pacman -S fcitx5 fcitx5-im fcitx5-qt fcitx5-gtk fcitx5-chinese-addons fcitx5-configtool fcitx5-rime 
```

```bash
# ~/.xprofile
GTK_IM_MODULE DEFAULT=fcitx
QT_IM_MODULE  DEFAULT=fcitx
XMODIFIERS    DEFAULT=\@im=fcitx
INPUT_METHOD  DEFAULT=fcitx
SDL_IM_MODULE DEFAULT=fcitx
GLFW_IM_MODULE DEFAULT=ibus
```


首先检查一下 locale 配置，`locale -a` 看一下结果中是否有 `zh_CN.utf8`，如果没有请先修改 `/etc/locale.gen` 文件将 `zh_CN.utf8` 取消注释，然后使用 `sudo locale-gen` 重新生成。

```bash
export GTK_IM_MODULE=fcitx
export QT_IM_MODULE=fcitx
export XIM=fcitx
export XIM_PROGRAM=fcitx
export XMODIFIERS="@im=fcitx"
export SDL_IM_MODULE=fcitx
export LC_CTYPE=zh_CN.UTF-8
```


## 安装Clash
国内下载较慢的话，可以使用大佬提供的代理，使用方式是在下载链接的前面加上 `https://ghproxy.com/` 即可，则代理后的链接为`https://ghproxy.com/https://github.com/Fndroid/clash_for_windows_pkg/releases/download/0.20.34/Clash.for.Windows-0.20.34-x64-linux.tar.gz`

然后解压：`tar -zxvf Clash.for.Windows.xxx` 

```bash
mv xxxx ~/.local/share
mkdir -p ~/.local/bin
ln -s /home/${USER}/.local/xxxx/cfw /home/${USER}/.local/bin/cfw # 软链接
export PATH=/home/${USER}/.local/bin:$PATH # 环境变量
```

```bash
sudo vim ~/.local/share/applications/clash.desktop

# 写入如下内容
[Desktop Entry]
Name=Clash Fow Windows
Exec=/home/user/.local/bin/cfw
Icon=/home/user/.local/bin/cfw
Type=Application
StartupNotify=true

# 设置权限
sudo chmod +x ~/.local/share/applications/clash.desktop
```





