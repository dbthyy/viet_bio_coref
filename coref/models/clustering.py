class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px != py:
            self.parent[py] = px

def union_find_clustering(scores, threshold=0.5):
    N = scores.size(0)
    uf = UnionFind(N)

    for i in range(N):
        for j in range(i + 1, N):
            if scores[i, j] >= threshold:
                uf.union(i, j)

    clusters_dict = {}
    for i in range(N):
        root = uf.find(i)
        if root not in clusters_dict:
            clusters_dict[root] = set()
        clusters_dict[root].add(i)

    return list(clusters_dict.values())