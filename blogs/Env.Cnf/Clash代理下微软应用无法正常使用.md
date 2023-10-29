---
title: Clash代理下微软应用无法正常使用
date: 2023/09/29 11:27:03
categories:
    - 环境配置
tags: 
    - Clash
    - Windows
---
Clash代理下，微软的OneDrive、Store无法使用

<!-- more -->

本质上是因为 微软 uwp 应用默认直连，所以开了代理就用不了

【最终解决方案】
无脑批量解除所有UWP应用限制（管理员）
```powershell
Get-ChildItem -Path Registry::"HKCU\Software\Classes\Local Settings\Software\Microsoft\Windows\CurrentVersion\AppContainer\Mappings\" -name | ForEach-Object {CheckNetIsolation.exe LoopbackExempt -a -p="$_"}
```