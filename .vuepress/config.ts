import { defineUserConfig } from "vuepress";
import recoTheme from "vuepress-theme-reco";
import { mdEnhancePlugin } from "vuepress-plugin-md-enhance";

export default defineUserConfig({
  title: "真真夜夜的博客",
  description: "Just playing around",
  theme: recoTheme({
    // 主题颜色
    colorMode: 'dark',
    colorModeSwitch: false,
    style: "@vuepress-reco/style-default",
    logo: "/avatar.png",
    author: "真真夜夜",
    authorAvatar: "/avatar2.png",
    autoSetSeries: true,

    head: [
      [
        "link",
        {
          rel: "stylesheet",
          href: "https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css",
        },
      ], // 让md支持数学公式
    ],


    navbar: [
      { text: "Home", link: "/" },
      { text: "Categories", link: "/categories/jiqixuexi/1/" },
      { text: "Tags", link: "/tags/Archlinux/1/" },
      {
        text: '文档',
        icon: 'SubVolume',
        children: [
          // { text: '论文阅读', link: '/docs/Papers/强化学习/2.md' },
          { text: 'LeetCode', link: '/docs/LeetCode/88.md'},
        ],
      },
      { text: "About", link: "/blogs/Other/about.html"}
    ],
  }),

  plugins: [
    mdEnhancePlugin({
      // 使用 KaTeX 启用 TeX 支持
      katex: false,
      // 使用 mathjax 启用 TeX 支持
      mathjax: true,
    }),
  ],

});

