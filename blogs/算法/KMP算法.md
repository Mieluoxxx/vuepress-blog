---
title: KMP算法（next数组的构建）
date: 2023/11/16 22:28:37
categories:
    - 算法
tags: 
    - KMP
---

对于字符串“abababca”，它的next如下表所示：

![图片.png](https://s2.loli.net/2023/11/16/zxAe9q4uGwP3ndO.png)

```cpp
class Solution {	// 不设置哨兵
public:
    int strStr(string s, string p) {
        int n = s.size(), m = p.size();
        if (m == 0) return 0;
        
        vector<int> next(m);
        for (int i = 1, j = 0; i < m; i++) {
          	// j > 0 可以让next[0]=0
            while (j > 0 && p[i] != p[j]) j = next[j - 1];	
            if (p[i] == p[j]) j++;
            next[i] = j;
        }

        for (int i = 0, j = 0; i < n; i++) {
            while (j > 0 && s[i] != p[j]) j = next[j - 1];
            if (s[i] == p[j]) j++;
            if (j == m) return i - m + 1;
        }
        return -1;
    }
};
```

