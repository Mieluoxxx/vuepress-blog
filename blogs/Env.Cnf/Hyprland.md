---
title: Hyprland：Linux最终DE
date: 2023/10/23 20:49:00
categories:
    - 环境配置
tags: 
    - Archlinux
    - Hyprland
sticky: 1
---

## 解决Xwayland应用模糊问题（such as QQ, jetbrains）
```shell
# 先禁用XWayland的缩放
# unscale XWayland
xwayland {
  force_zero_scaling = true
}
# toolkit-specific scale`
env = GDK_SCALE,2
env = XCURSOR_SIZE,32
```

## 解决Electron在Wayland下的模糊（Vscode...）
```shell
# 第一种
<app executable> --enable-features=UseOzonePlatform --ozone-platform=wayland

# 第二种在 ~/.config/code-flags.conf 中加上两行：
```shell
--enable-features=WaylandWindowDecorations
--ozone-platform-hint=auto
```
```
