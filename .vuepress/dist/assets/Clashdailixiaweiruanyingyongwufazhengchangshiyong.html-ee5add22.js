import{_ as a,o as n,c as s,f as e,a as o,e as t}from"./app-0978fb63.js";const p={},c=o("p",null,"Clash代理下，微软的OneDrive、Store无法使用",-1),i=t(`<p>本质上是因为 微软 uwp 应用默认直连，所以开了代理就用不了</p><p>【最终解决方案】 无脑批量解除所有UWP应用限制（管理员）</p><div class="language-powershell line-numbers-mode" data-ext="powershell"><pre class="language-powershell"><code><span class="token function">Get-ChildItem</span> <span class="token operator">-</span>Path Registry::<span class="token string">&quot;HKCU\\Software\\Classes\\Local Settings\\Software\\Microsoft\\Windows\\CurrentVersion\\AppContainer\\Mappings\\&quot;</span> <span class="token operator">-</span>name <span class="token punctuation">|</span> <span class="token function">ForEach-Object</span> <span class="token punctuation">{</span>CheckNetIsolation<span class="token punctuation">.</span>exe LoopbackExempt <span class="token operator">-</span>a <span class="token operator">-</span>p=<span class="token string">&quot;<span class="token variable">$_</span>&quot;</span><span class="token punctuation">}</span>
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div></div></div>`,3);function l(r,u){return n(),s("div",null,[c,e(" more "),i])}const h=a(p,[["render",l],["__file","Clashdailixiaweiruanyingyongwufazhengchangshiyong.html.vue"]]);export{h as default};