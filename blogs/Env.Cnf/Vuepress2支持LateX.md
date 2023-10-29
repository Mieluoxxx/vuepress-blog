---
title: Vuepress2支持LateX.md
date: 2023/10/29 10:38:32
categories:
    - 环境配置
tags: 
    - Blog
---

在你的项目中安装相关的Tex包 katex 或 mathjax-full:
```bash
yarn add -D katex
# or
yarn add -D mathjax-full
```
之后启动它
```ts
// .vuepress/config.ts
import { mdEnhancePlugin } from "vuepress-plugin-md-enhance";

export default {
  plugins: [
    mdEnhancePlugin({
      // 使用 KaTeX 启用 TeX 支持
      katex: true,
      // 使用 mathjax 启用 TeX 支持
      mathjax: true,
    }),
  ],
};
```
::: warning
你只能启用其中一个，并且 katex 具有更高的优先级。
:::

```tex
Euler's identity $e^{i\pi}+1=0$ is a beautiful formula in $\mathbb{R}^2$.
```
Euler's identity $e^{i\pi}+1=0$ is a beautiful formula in $\mathbb{R}^2$.

```tex
$$
\frac {\partial^r} {\partial \omega^r} \left(\frac {y^{\omega}} {\omega}\right)
= \left(\frac {y^{\omega}} {\omega}\right) \left\{(\log y)^r + \sum_{i=1}^r \frac {(-1)^i r \cdots (r-i+1) (\log y)^{r-i}} {\omega^i} \right\}
$$
```
$$
\frac {\partial^r} {\partial \omega^r} \left(\frac {y^{\omega}} {\omega}\right)
= \left(\frac {y^{\omega}} {\omega}\right) \left\{(\log y)^r + \sum_{i=1}^r \frac {(-1)^i r \cdots (r-i+1) (\log y)^{r-i}} {\omega^i} \right\}
$$


[更多Markdown增强](https://plugin-md-enhance.vuejs.press/zh/)