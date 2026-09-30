import matplotlib
matplotlib.use('TkAgg')  # 改成系统自带的 GUI 后端
import matplotlib.pyplot as plt
import numpy as np
def plot_merged_graph(merged_g, k0, n, k):
    X = merged_g.ndata['feat'].cpu().numpy()
    src, dst = merged_g.edges()
    src = src.cpu().numpy()
    dst = dst.cpu().numpy()

    NA = k0 + n * k
    NB = merged_g.num_nodes() - NA

    plt.figure(figsize=(6,6))
    plt.title("Merged Graph (A=red, B=blue)")

    # 边
    for s, d in zip(src, dst):
        x1, y1 = X[s]
        x2, y2 = X[d]
        plt.plot([x1, x2], [y1, y2], 'gray', alpha=0.3)

    # A节点 (红)
    plt.scatter(X[:NA, 0], X[:NA, 1], c='red', s=30, label='A')
    # B节点 (蓝)
    plt.scatter(X[NA:, 0], X[NA:, 1], c='blue', s=30, label='B')

    plt.legend()
    plt.axis('equal')
    plt.show()


def plot_merged_graph_intensity(merged_g, k0, n, k, title="Merged Graph (A=red, B=blue, Link=gray)"):
    """
    可视化 merged 图：A红，B蓝，A-B连接灰，重复边加深。
    """
    X = merged_g.ndata['feat'].cpu().numpy()
    src, dst = merged_g.edges()
    src = src.cpu().numpy()
    dst = dst.cpu().numpy()

    # 区分节点范围
    NA = k0 + n * k
    NB = merged_g.num_nodes() - NA

    # 统计重复边（无向合并）
    edge_counts = {}
    for s, d in zip(src, dst):
        key = tuple(sorted((int(s), int(d))))
        edge_counts[key] = edge_counts.get(key, 0) + 1

    # 归一化计数
    counts = np.array(list(edge_counts.values()))
    cmin, cmax = counts.min(), counts.max()

    def lw(c):
        if cmax == cmin:
            return 1.0
        return 1.0 + 5.0 * (np.sqrt(c - cmin) / np.sqrt(cmax - cmin))

    def alpha(c):
        if cmax == cmin:
            return 0.3
        return min(1.0, 0.15 + 0.7 * (c - cmin) / (cmax - cmin))

    plt.figure(figsize=(7, 7))
    plt.title(title)

    # 绘制边
    for (u, v), c in edge_counts.items():
        x1, y1 = X[u]
        x2, y2 = X[v]
        lw_val = lw(c)
        a_val = alpha(c)

        # 分类边类型：A内部、B内部、AB连接
        if u < NA and v < NA:
            color = 'red'
        elif u >= NA and v >= NA:
            color = 'blue'
        else:
            color = 'gray'
        plt.plot([x1, x2], [y1, y2], color=color, linewidth=lw_val, alpha=a_val)

    # 绘制节点
    plt.scatter(X[:NA, 0], X[:NA, 1], c='red', s=30, label='A-nodes')
    plt.scatter(X[NA:, 0], X[NA:, 1], c='blue', s=30, label='B-nodes')
    # ✅ 标注节点 ID
    for i, (x, y) in enumerate(X):
        plt.text(x, y, str(i), fontsize=18, color='b', ha='center', va='center')


    plt.legend()
    plt.axis('equal')
    plt.tight_layout()
    plt.show()