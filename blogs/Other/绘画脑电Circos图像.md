---
title: 安装并使用Brainconn绘制Circos图像
date: 2023/05/26 00:50:00
categories:
    - 科研
tags: 
    - 脑科学
---

使用Python-MNE绘制脑电Circos图像

<!-- more -->

### 绘画脑电Circos图像（Python）

```python
import numpy as np
from mne_connectivity.viz import plot_connectivity_circle
import matplotlib.pyplot as plt
from mne.viz import circular_layout

fig, ax = plt.subplots(figsize=(8, 8), facecolor='black',
                       subplot_kw=dict(polar=True))
conmat = [[0.00, 0.28, 0.56, 0.40, 0.52, 0.57, 0.92, 0.68, 0.65, 0.47],
          [0.28, 0.00, 0.17, 0.27, 0.98, 0.32, 0.46, 0.92, 0.57, 0.87],
          [0.56, 0.17, 0.00, 0.62, 0.95, 0.11, 0.43, 0.60, 0.67, 0.63],
          [0.40, 0.27, 0.62, 0.00, 0.98, 0.25, 0.78, 0.41, 0.19, 0.58],
          [0.52, 0.98, 0.95, 0.98, 0.00, 0.20, 0.92, 0.26, 0.70, 0.96],
          [0.57, 0.32, 0.11, 0.25, 0.20, 0.00, 0.62, 0.57, 0.49, 0.82],
          [0.92, 0.46, 0.43, 0.78, 0.92, 0.62, 0.00, 0.70, 0.85, 0.94],
          [0.68, 0.92, 0.60, 0.41, 0.26, 0.57, 0.70, 0.00, 0.55, 0.94],
          [0.65, 0.57, 0.67, 0.19, 0.70, 0.49, 0.85, 0.55, 0.00, 0.98],
          [0.47, 0.87, 0.63, 0.58, 0.96, 0.82, 0.94, 0.94, 0.98, 0.00]]

conmat = np.array(conmat)

nodes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

node_angles = circular_layout(nodes, nodes, start_pos=90,
                              group_boundaries=[0, len(nodes) / 2])

plot_connectivity_circle(conmat, nodes, n_lines=20, node_angles=node_angles,title='使用mne绘制', ax=ax)
```

![Circos](https://s2.loli.net/2023/07/28/TX5OoQpkzr631v7.png)
