import{_ as n,o as s,c as a,f as e,a as t,e as i}from"./app-0978fb63.js";const o={},l=t("p",null,"真是奇怪的需求，我的建议是，全部重装得了",-1),c=i(`<ul><li>首先将新硬盘分区</li></ul><div class="language-bash line-numbers-mode" data-ext="sh"><pre class="language-bash"><code><span class="token function">mount</span> /dev/nvme1n1p2 /mnt  <span class="token comment">#挂载分区</span>
<span class="token function">rsync</span> <span class="token parameter variable">-aAXv</span> /* /mnt <span class="token parameter variable">--exclude</span><span class="token operator">=</span><span class="token punctuation">{</span><span class="token string">&quot;/dev/*&quot;</span>,<span class="token string">&quot;/proc/*&quot;</span>,<span class="token string">&quot;/sys/*&quot;</span>,<span class="token string">&quot;/tmp/*&quot;</span>,<span class="token string">&quot;/run/*&quot;</span>,<span class="token string">&quot;/mnt/*&quot;</span>,<span class="token string">&quot;/media/*&quot;</span>,<span class="token string">&quot;/lost+found&quot;</span>,<span class="token string">&quot;/boot&quot;</span><span class="token punctuation">}</span> <span class="token comment">#复制文件</span>
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div></div></div><ul><li>用 Archiso修复引导</li></ul><div class="language-bash line-numbers-mode" data-ext="sh"><pre class="language-bash"><code>iwctl station   <span class="token comment"># 联网</span>
wlan0 get-networks station 
wlan0 connect xxx
<span class="token function">ping</span> baidu.com
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><div class="language-bash line-numbers-mode" data-ext="sh"><pre class="language-bash"><code><span class="token function">mount</span> /dev/nvme0n1p2 /mnt <span class="token comment">#挂载新SSD的根分区到/mnt</span>
<span class="token function">mkdir</span> /mnt/boot
<span class="token function">mount</span> /dev/nvme0n1p1 /mnt/boot
genfstab <span class="token parameter variable">-L</span> /mnt <span class="token operator">&gt;</span> /mnt/etc/fstab  <span class="token comment">#生成filesystem table</span>
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><div class="language-bash line-numbers-mode" data-ext="sh"><pre class="language-bash"><code>arch-chroot /mnt               
pacman <span class="token parameter variable">-S</span> grub  <span class="token comment">#安装依赖包</span>
pacman <span class="token parameter variable">-S</span> linux <span class="token comment">#重装linux</span>
grub-install <span class="token parameter variable">--target</span><span class="token operator">=</span>x86_64-efi --efi-directory<span class="token operator">=</span>/boot --bootloader-id<span class="token operator">=</span>grub
<span class="token function">grub-mkconfig</span> <span class="token parameter variable">-o</span> /boot/grub/grub.cfg
<span class="token comment"># 检查 /boot/grub/grub.cfg 和 /mnt/etc/fstab中的UUID是否对应，同时还需要使用lsblk来查看目前分区的UUID</span>
<span class="token builtin class-name">exit</span>
<span class="token function">reboot</span>
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div>`,6);function r(p,u){return s(),a("div",null,[l,e(" more "),c])}const m=n(o,[["render",r],["__file","Archlinuxqianyidaoxingutai.html.vue"]]);export{m as default};