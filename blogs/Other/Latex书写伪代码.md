---
title: Latex书写伪代码
date: 2023/12/03 17:26:00
categories:
    - 论文写作
tags: 
    - Latex
---

编译器选择XeLaTeX

```latex
\documentclass{article}
%\usepackage{algorithm2e}
\usepackage[ruled,longend,linesnumbered]{algorithm2e}
\usepackage{xeCJK}
\begin{document}

\begin{algorithm}
% \DontPrintSemicolon
% \KwData{}
% \KwResult{}
\KwIn{时间}
\KwOut{知识}

\Begin{
我在B站刷到一个视频\;
这个视频似乎对我有些帮助\;
    \While{video is playing}
    {
        继续观看\;
        \eIf{understand}
        {
            看下部分\;
            下部分变为这部分\;
        }
        {
            回看这部分\;
        }
    }
    我学会了也不给三连
}

\caption{如何生成好看的伪代码}
\end{algorithm}
\end{document}
```

![1701595651547.png](http://pic.moguw.top/i/2023/12/03/656c4a067ad0d.png)

