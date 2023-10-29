import{_ as n,o as i,c as e,e as s}from"./app-0978fb63.js";const d={},l=s(`<p>给你两个按 <strong>非递减顺序</strong> 排列的整数数组 <code>nums1</code> 和 <code>nums2</code>，另有两个整数 <code>m</code> 和 <code>n</code> ，分别表示 <code>nums1</code> 和 <code>nums2</code> 中的元素数目。</p><p>请你 <strong>合并</strong> <code>nums2</code> 到 <code>nums1</code> 中，使合并后的数组同样按 <strong>非递减顺序</strong> 排列。</p><p><strong>注意</strong>：最终，合并后数组不应由函数返回，而是存储在数组 <code>nums1</code> 中。为了应对这种情况，<code>nums1</code> 的初始长度为 <code>m + n</code>，其中前 <code>m</code> 个元素表示应合并的元素，后 <code>n</code> 个元素为 <code>0</code> ，应忽略。<code>nums2</code> 的长度为 <code>n</code> 。</p><p><strong>示例 1：</strong></p><div class="language-Java line-numbers-mode" data-ext="Java"><pre class="language-Java"><code>输入：nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
输出：[1,2,2,3,5,6]
解释：需要合并 [1,2,3] 和 [2,5,6] 。
合并结果是 [1,2,2,3,5,6] ，其中斜体加粗标注的为 nums1 中的元素。
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><p><strong>示例 2：</strong></p><div class="language-Java line-numbers-mode" data-ext="Java"><pre class="language-Java"><code>输入：nums1 = [1], m = 1, nums2 = [], n = 0
输出：[1]
解释：需要合并 [1] 和 [] 。
合并结果是 [1] 。
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><p><strong>示例 3：</strong></p><div class="language-Java line-numbers-mode" data-ext="Java"><pre class="language-Java"><code>输入：nums1 = [0], m = 0, nums2 = [1], n = 1
输出：[1]
解释：需要合并的数组是 [] 和 [1] 。
合并结果是 [1] 。
注意，因为 m = 0 ，所以 nums1 中没有元素。nums1 中仅存的 0 仅仅是为了确保合并结果可以顺利存放到 nums1 中。
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><p><strong>提示：</strong></p><ul><li><code>nums1.length == m + n</code></li><li><code>nums2.length == n</code></li><li><code>0 &lt;= m, n &lt;= 200</code></li><li><code>1 &lt;= m + n &lt;= 200</code></li><li><code>-109 &lt;= nums1[i], nums2[j] &lt;= 109</code><strong>进阶</strong>：你可以设计实现一个时间复杂度为 <code>O(m + n)</code> 的算法解决此问题吗？</li></ul><div class="language-C++ line-numbers-mode" data-ext="C++"><pre class="language-C++"><code>class Solution {
public:
    void merge(vector&lt;int&gt;&amp; nums1, int m, vector&lt;int&gt;&amp; nums2, int n) {
      for(int i = 0; i != n; ++i){
          nums1[n+m] = nums2[i];
      }
      sort(nums1.begin(), nums2.end());
    }
};
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><div class="language-Python line-numbers-mode" data-ext="Python"><pre class="language-Python"><code>class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -&gt; None:
        &quot;&quot;&quot;
        Do not return anything, modify nums1 in-place instead.
        &quot;&quot;&quot;
        for i in range(n):
            nums1[m+i] = nums2[i]
        nums1.sort()
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><div class="language-C++ line-numbers-mode" data-ext="C++"><pre class="language-C++"><code>class Solution {
public:
    void merge(vector&lt;int&gt;&amp; nums1, int m, vector&lt;int&gt;&amp; nums2, int n) {
        int p1 = 0, p2 = 0;
        int cur = 0;
        int sorted[m+n];
        while(p1 &lt; m || p2 &lt; n){
            if (p1 == m)
                cur = nums2[p2++];
            else if (p2 == n)
                cur = nums1[p1++];
            else if (nums1[p1] &lt; nums2[p2])
                cur = nums1[p1++];
            else
                cur = nums2[p2++];
            sorted[p1+p2-1] = cur;
        }
        for(int i = 0; i != m+n; i++)
            nums1[i] = sorted[i];
    }
};
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><div class="language-Plain line-numbers-mode" data-ext="Plain"><pre class="language-Plain"><code>class Solution:
    def merge(self, nums1: List[int], m: int, nums2: List[int], n: int) -&gt; None:
        &quot;&quot;&quot;
        Do not return anything, modify nums1 in-place instead.
        &quot;&quot;&quot;
        sorted = []
        p1, p2 = 0, 0
        while p1 &lt; m or p2 &lt; n:
            if p1 == m:
                sorted.append(nums2[p2])
                p2 += 1
            elif p2 == n:
                sorted.append(nums1[p1])
                p1 += 1
            elif nums1[p1] &lt; nums2[p2]:
                sorted.append(nums1[p1])
                p1 += 1
            else:
                sorted.append(nums2[p2])
                p2 += 1
        nums1[:] = sorted
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><div class="language-C++ line-numbers-mode" data-ext="C++"><pre class="language-C++"><code>class Solution {
public:
    void merge(vector&lt;int&gt;&amp; nums1, int m, vector&lt;int&gt;&amp; nums2, int n) {
        int i = m - 1;
        int j = n - 1;
        int k = m + n - 1;
        while (j &gt;= 0) {
            if (i &gt;= 0 &amp;&amp; nums1[i] &gt; nums2[j]) {
                nums1[k--] = nums1[i--];
            } else {
                nums1[k--] = nums2[j--];
            }
        }
    }
};
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div><div class="language-C++ line-numbers-mode" data-ext="C++"><pre class="language-C++"><code>class Solution {
public:
    void merge(vector&lt;int&gt;&amp; nums1, int m, vector&lt;int&gt;&amp; nums2, int n) {
        while(n) nums1[m+n] = (m==0 || nums1[m-1]&lt;nums2[n-1]) ? nums2[--n] : nums1[--m];
    }
};
</code></pre><div class="line-numbers" aria-hidden="true"><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div><div class="line-number"></div></div></div>`,17),a=[l];function v(m,u){return i(),e("div",null,a)}const r=n(d,[["render",v],["__file","88.html.vue"]]);export{r as default};
