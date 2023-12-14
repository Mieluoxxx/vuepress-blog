---
title: Floyd
date: 2023/12/06
categories:
  - 算法
tags
  - 最短路
---

```python
from typing import List

def floyd(graph: List[List[float]]) -> List[List[float]]:
    nodes = len(graph)
    
    # 初始化距离矩阵，dist[i][j]表示节点i到节点j的距离
    dist = [[float('inf') for _ in range(nodes)] for _ in range(nodes)]
    for i in range(nodes):
        for j in range(nodes):
            dist[i][j] = graph[i][j]
    
    # 更新距离矩阵
    for k in range(nodes):  # 经过k点
        for i in range(nodes):
            for j in range(nodes):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    
    return dist

def print_shortest_distances(distances: List[List[float]]) -> None:
    num_nodes = len(distances)
    
    print("最短路径距离:")
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and distances[i][j] != float('inf'):
                print(f"{i + 1} -> {j + 1}: {distances[i][j]}")


# 示例
graph: List[List[float]] = [
    [0, 3, float('inf'), 7],
    [8, 0, 2, float('inf')],
    [5, float('inf'), 0, 1],
    [2, float('inf'), float('inf'), 0]
]

graph2 = [
    [0, 3, 5, 2],
    [3, 0, 2, float('inf')],
    [5, 2, 0, 1],
    [2, float('inf'), 1, 0]
]

result: List[List[float]] = floyd(graph)
print_shortest_distances(result)

```

