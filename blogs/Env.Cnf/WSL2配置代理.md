---
title: WSL2配置代理
date: 2023/03/02 09:14:03
categories:
  - 环境配置
tags: 
  - Clash
---
WSL2（Debian/Ubuntu），Clash代理的解决方案

<!-- more -->
新建`proxy.sh`文件，内容如下：
```bash
#!/bin/sh
hostip=$(cat /etc/resolv.conf | grep nameserver | awk '{ print $2 }')
wslip=$(hostname -I | awk '{print $1}')
port=7890
 
PROXY_HTTP="http://${hostip}:${port}"
 
set_proxy(){
  export http_proxy="${PROXY_HTTP}"
  export HTTP_PROXY="${PROXY_HTTP}"
 
  export https_proxy="${PROXY_HTTP}"
  export HTTPS_proxy="${PROXY_HTTP}"
 
  export ALL_PROXY="${PROXY_SOCKS5}"
  export all_proxy=${PROXY_SOCKS5}
 
  git config --global http.https://github.com.proxy ${PROXY_HTTP}
  git config --global https.https://github.com.proxy ${PROXY_HTTP}
 
  echo "Proxy has been opened."
}
 
unset_proxy(){
  unset http_proxy
  unset HTTP_PROXY
  unset https_proxy
  unset HTTPS_PROXY
  unset ALL_PROXY
  unset all_proxy
  git config --global --unset http.https://github.com.proxy
  git config --global --unset https.https://github.com.proxy
 
  echo "Proxy has been closed."
}
 
test_setting(){
  echo "Host IP:" ${hostip}
  echo "WSL IP:" ${wslip}
  echo "Try to connect to Google..."
  resp=$(curl -I -s --connect-timeout 5 -m 5 -w "%{http_code}" -o /dev/null www.google.com)
  if [ ${resp} = 200 ]; then	
    echo "Proxy setup succeeded!"
  else
    echo "Proxy setup failed!"
  fi
}
 
if [ "$1" = "set" ]
then
  set_proxy
 
elif [ "$1" = "unset" ]
then
  unset_proxy
 
elif [ "$1" = "test" ]
then
  test_setting
else
  echo "Unsupported arguments."
fi
```
- source ./proxy.sh set：开启代理
- source ./proxy.sh unset：关闭代理
- source ./proxy.sh test：查看代理状态

### 第四步对任意路径开启代理
在`~/.zshrc`中添加
`alias proxy="source ~/proxy.sh"`
刷新环境变量
`source ~/.zshrc`
- proxy set：开启代理
- proxy unset：关闭代理
- proxy test：查看代理状态
  
### 第五步自动添加代理（可选）
在`~/.zshrc`中加入`. ~/proxy.sh set`