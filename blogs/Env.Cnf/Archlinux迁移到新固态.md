---
title: Archlinux迁移到新固态
date: 2023/04/19 08:34:32
categories:
    - 环境配置
tags: 
    - Archlinux
---

真是奇怪的需求，我的建议是，全部重装得了

<!-- more -->

- 首先将新硬盘分区
```bash
mount /dev/nvme1n1p2 /mnt  #挂载分区
rsync -aAXv /* /mnt --exclude={"/dev/*","/proc/*","/sys/*","/tmp/*","/run/*","/mnt/*","/media/*","/lost+found","/boot"} #复制文件
```
- 用 Archiso修复引导
```bash
iwctl station   # 联网
wlan0 get-networks station 
wlan0 connect xxx
ping baidu.com
```
```bash
mount /dev/nvme0n1p2 /mnt #挂载新SSD的根分区到/mnt
mkdir /mnt/boot
mount /dev/nvme0n1p1 /mnt/boot
genfstab -L /mnt > /mnt/etc/fstab  #生成filesystem table
```
```bash
arch-chroot /mnt               
pacman -S grub  #安装依赖包
pacman -S linux #重装linux
grub-install --target=x86_64-efi --efi-directory=/boot --bootloader-id=grub
grub-mkconfig -o /boot/grub/grub.cfg
# 检查 /boot/grub/grub.cfg 和 /mnt/etc/fstab中的UUID是否对应，同时还需要使用lsblk来查看目前分区的UUID
exit
reboot
```
