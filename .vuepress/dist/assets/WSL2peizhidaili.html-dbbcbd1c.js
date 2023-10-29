import{_ as s,o as n,c as a,f as e,a as t,e as l}from"./app-0978fb63.js";const p={},i=t("p",null,"WSL2（Debian/Ubuntu），Clash代理的解决方案",-1),o=l(`<p>新建<code>proxy.sh</code>文件，内容如下：</p><div class="language-bash line-numbers-mode" data-ext="sh"><pre class="language-bash"><code><span class="token shebang important">#!/bin/sh</span>
<span class="token assign-left variable">hostip</span><span class="token operator">=</span><span class="token variable"><span class="token variable">$(</span><span class="token function">cat</span> /etc/resolv.conf <span class="token operator">|</span> <span class="token function">grep</span> nameserver <span class="token operator">|</span> <span class="token function">awk</span> <span class="token string">&#39;{ print $2 }&#39;</span><span class="token variable">)</span></span>
<span class="token assign-left variable">wslip</span><span class="token operator">=</span><span class="token variable"><span class="token variable">$(</span><span class="token function">hostname</span> <span class="token parameter variable">-I</span> <span class="token operator">|</span> <span class="token function">awk</span> <span class="token string">&#39;{print $1}&#39;</span><span class="token variable">)</span></span>
<span class="token assign-left variable">port</span><span class="token operator">=</span><span class="token number">7890</span>
 
<span class="token assign-left variable">PROXY_HTTP</span><span class="token operator">=</span><span class="token string">&quot;http://<span class="token variable">\${hostip}</span>:<span class="token variable">\${port}</span>&quot;</span>
 
<span class="token function-name function">set_proxy</span><span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">{</span>
  <span class="token builtin class-name">export</span> <span class="token assign-left variable">http_proxy</span><span class="token operator">=</span><span class="token string">&quot;<span class="token variable">\${PROXY_HTTP}</span>&quot;</span>
  <span class="token builtin class-name">export</span> <span class="token assign-left variable">HTTP_PROXY</span><span class="token operator">=</span><span class="token string">&quot;<span class="token variable">\${PROXY_HTTP}</span>&quot;</span>
 
  <span class="token builtin class-name">export</span> <span class="token assign-left variable">https_proxy</span><span class="token operator">=</span><span class="token string">&quot;<span class="token variable">\${PROXY_HTTP}</span>&quot;</span>
  <span class="token builtin class-name">export</span> <span class="token assign-left variable">HTTPS_proxy</span><span class="token operator">=</span><span class="token string">&quot;<span class="token variable">\${PROXY_HTTP}</span>&quot;</span>
 
  <span class="token builtin class-name">export</span> <span class="token assign-left variable">ALL_PROXY</span><span class="token operator">=</span><span class="token string">&quot;<span class="token variable">\${PROXY_SOCKS5}</span>&quot;</span>
  <span class="token builtin class-name">export</span> <span class="token assign-left variable">all_proxy</span><span class="token operator">=</span><span class="token variable">\${PROXY_SOCKS5}</span>
 
  <span class="token function">git</span> config <span class="token parameter variable">--global</span> http.https://github.com.proxy <span class="token variable">\${PROXY_HTTP}</span>
  <span class="token function">git</span> config <span class="token parameter variable">--global</span> https.https://github.com.proxy <span class="token variable">\${PROXY_HTTP}</span>
 
  <span class="token builtin class-name">echo</span> <span class="token string">&quot;Proxy has been opened.&quot;</span>
<span class="token punctuation">}</span>
 
<span class="token function-name function">unset_proxy</span><span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">{</span>
  <span class="token builtin class-name">unset</span> http_proxy
  <span class="token builtin class-name">unset</span> HTTP_PROXY
  <span class="token builtin class-name">unset</span> https_proxy
  <span class="token builtin class-name">unset</span> HTTPS_PROXY
  <span class="token builtin class-name">unset</span> ALL_PROXY
  <span class="token builtin class-name">unset</span> all_proxy
  <span class="token function">git</span> config <span class="token parameter variable">--global</span> <span class="token parameter variable">--unset</span> http.https://github.com.proxy
  <span class="token function">git</span> config <span class="token parameter variable">--global</span> <span class="token parameter variable">--unset</span> https.https://github.com.proxy
 
  <span class="token builtin class-name">echo</span> <span class="token string">&quot;Proxy has been closed.&quot;</span>
<span class="token punctuation">}</span>
 
<span class="token function-name function">test_setting</span><span class="token punctuation">(</span><span class="token punctuation">)</span><span class="token punctuation">{</span>
  <span class="token builtin class-name">echo</span> <span class="token string">&quot;Host IP:&quot;</span> <span class="token variable">\${hostip}</span>
  <span class="token builtin class-name">echo</span> <span class="token string">&quot;WSL IP:&quot;</span> <span class="token variable">\${wslip}</span>
  <span class="token builtin class-name">echo</span> <span class="token string">&quot;Try to connect to Google...&quot;</span>
  <span class="token assign-left variable">resp</span><span class="token operator">=</span><span class="token variable"><span class="token variable">$(</span><span class="token function">curl</span> <span class="token parameter variable">-I</span> <span class="token parameter variable">-s</span> --connect-timeout <span class="token number">5</span> <span class="token parameter variable">-m</span> <span class="token number">5</span> <span class="token parameter variable">-w</span> <span class="token string">&quot;%{http_code}&quot;</span> <span class="token parameter variable">-o</span> /dev/null www.google.com<span class="token variable">)</span></span>
  <span class="token keyword">if</span> <span class="token punctuation">[</span> <span class="token variable">\${resp}</span> <span class="token operator">=</span> <span class="token number">200</span> <span class="token punctuation">]</span><span class="token punctuation">;</span> <span class="token keyword">then</span>	
    <span class="token builtin class-name">echo</span> <span class="token string">&quot;Proxy setup succeeded!&quot;</span>
  <span class="token keyword">else</span>
    <span class="token builtin class-name">echo</span> <span class="token string">&quot;Proxy setup failed!&quot;</span>
  <span class="token keyword">fi</span>
<span class="token punctuation">}</span>
 
<span class="token keyword">if</span> <span class="token punctuation">[</span> <span class="token string">&quot;<span class="token variable">$1</span>&quot;</span> <span class="token operator">=</span> <span class="token string">&quot;set&quot;</span> <span class="token punctuation">]</span>
<span class="token keyword">then</span>
  set_proxy
 
<span class="token keyword">elif</span> <span class="token punctuation">[</span> <span class="token string">&quot;<span class="token variable">$1</span>&quot;</span> <span class="token operator">=</span> <span class="token string">&quot;unset&quot;</span> <span class="token punctuation">]</span>
<span class="token keyword">then</span>
  unset_proxy
 
<span class="token keyword">elif</span> <span class="token punctuation">[</span> <span class="token string">&quot;<span class="token variable">$1</span>&quot;</span> <span class="token operator">=</span> <span class="token string">&quot;test&quot;</span> <span class="token punctuation">]</span>
<span class="token keyword">then</span>
  test_setting
<span class="token keyword">else</span>
  <span class="token builtin class-name">echo</span> <span class="token string">&quot;Unsupported arguments.&quot;</span>
<span class="token keyword">fi</span>
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><ul><li>source ./proxy.sh set：开启代理</li><li>source ./proxy.sh unset：关闭代理</li><li>source ./proxy.sh test：查看代理状态</li></ul><h3 id="第四步对任意路径开启代理" tabindex="-1"><a class="header-anchor" href="#第四步对任意路径开启代理" aria-hidden="true">#</a> 第四步对任意路径开启代理</h3><p>在<code>~/.zshrc</code>中添加 <code>alias proxy=&quot;source ~/proxy.sh&quot;</code> 刷新环境变量 <code>source ~/.zshrc</code></p><ul><li>proxy set：开启代理</li><li>proxy unset：关闭代理</li><li>proxy test：查看代理状态</li></ul><h3 id="第五步自动添加代理-可选" tabindex="-1"><a class="header-anchor" href="#第五步自动添加代理-可选" aria-hidden="true">#</a> 第五步自动添加代理（可选）</h3><p>在<code>~/.zshrc</code>中加入<code>. ~/proxy.sh set</code></p>`,8);function c(r,u){return n(),a("div",null,[i,e(" more "),o])}const v=s(p,[["render",c],["__file","WSL2peizhidaili.html.vue"]]);export{v as default};
