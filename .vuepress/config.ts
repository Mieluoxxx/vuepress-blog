import { defineUserConfig } from "vuepress";
import recoTheme from "vuepress-theme-reco";
import { mdEnhancePlugin } from "vuepress-plugin-md-enhance";

export default defineUserConfig({
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
      { text: "Home", link: "/", icon: "Home" },
      { text: "Categories", link: "/categories/jiqixuexi/1/", icon: "Category" },
      { text: "Tags", link: "/tags/Archlinux/1/", icon: "Tag" },
      { text: "时间轴", link: "/timeline", icon: "AlignHorizontalCenter" },
      {
        text: '文档',
        icon: 'DocumentAttachment',
        children: [
          { 
            text: '语言学习',
            children: [
              { text: 'Go', link: '/docs/Go/' },
              { text: 'Rust', link: '/docs/Rust/' },
            ]
          },
          {
            text: '框架学习',
            children: [
              { text: 'JAX', link: '/docs/JAX/' },
            ]
          },
          { 
            text: '实战',
            children: [
              { text: 'LeetCode', link: '/docs/LeetCode/' },
              { text: '时间序列', link: '/docs/TimeSeries/' },
            ]
          }
        ],
      },
      { text: "About", link: "/about/about.md", icon:"LocationHeart" }
    ],
  }),

  plugins: [
    mdEnhancePlugin({
      katex: false,
      mathjax: true,
    }),
  ],
  

});