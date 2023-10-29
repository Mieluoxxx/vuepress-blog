import{_ as a,o as n,c as e,e as s}from"./app-0978fb63.js";const l={},i=s(`<h2 id="解决xwayland应用模糊问题-such-as-qq-jetbrains" tabindex="-1"><a class="header-anchor" href="#解决xwayland应用模糊问题-such-as-qq-jetbrains" aria-hidden="true">#</a> 解决Xwayland应用模糊问题（such as QQ, jetbrains）</h2><div class="language-bash line-numbers-mode" data-ext="sh"><pre class="language-bash"><code><span class="token comment"># 先禁用XWayland的缩放</span>
<span class="token comment"># unscale XWayland</span>
xwayland <span class="token punctuation">{</span>
  force_zero_scaling <span class="token operator">=</span> <span class="token boolean">true</span>
<span class="token punctuation">}</span>
<span class="token comment"># toolkit-specific scale\`</span>
<span class="token function">env</span> <span class="token operator">=</span> GDK_SCALE,2
<span class="token function">env</span> <span class="token operator">=</span> XCURSOR_SIZE,32
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><h2 id="解决electron在wayland下的模糊-vscode" tabindex="-1"><a class="header-anchor" href="#解决electron在wayland下的模糊-vscode" aria-hidden="true">#</a> 解决Electron在Wayland下的模糊（Vscode...）</h2><div class="language-bash line-numbers-mode" data-ext="sh"><pre class="language-bash"><code><span class="token comment"># 第一种</span>
<span class="token operator">&lt;</span>app executable<span class="token operator">&gt;</span> --enable-features<span class="token operator">=</span>UseOzonePlatform --ozone-platform<span class="token operator">=</span>wayland

<span class="token comment"># 第二种在 ~/.config/code-flags.conf 中加上两行：</span>
\`\`\`shell
--enable-features<span class="token operator">=</span>WaylandWindowDecorations
--ozone-platform-hint<span class="token operator">=</span>auto
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><div class="language-text line-numbers-mode" data-ext="text"><pre class="language-text"><code></code></pre><div class="line-numbers" aria-hidden="true"></div></div>`,5),o=[i];function c(t,d){return n(),e("div",null,o)}const p=a(l,[["render",c],["__file","Hyprland.html.vue"]]);export{p as default};
