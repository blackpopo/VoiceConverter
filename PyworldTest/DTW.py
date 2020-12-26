import sys
import scipy
import numpy
class DTW:
    def __getstate__(self):
        d = self.__dict__.copy()
        if self.distance == self.cosine:
            d['distance'] = 'cosine'
        elif self.distance == self.euclidean:
            d['distance'] = 'euclidean'
        return d
    def __setstate__(self, dict):
        self.__dict__ = dict
        if dict['distance'] == 'cosine':
            self.distance = self.cosine
        elif dict['distance'] == 'euclidean':
            self.distance = self.euclidean
    def cosine(self, A, B):
        return scipy.dot(A, B.transpose()) / scipy.linalg.norm(A) \
                                                / scipy.linalg.norm(B)
    def euclidean(self, A, B):
        return scipy.linalg.norm(A - B)
    def __init__(self, source, target, distance = None, window = float('inf')):
        self.window = window
        self.source = source
        self.target = target
        if distance:
            self.distance = distance
        else:
            self.distance = self.euclidean
        self.dtw()
    def dtw(self):
        M, N = len(self.source), len(self.target)
        cost = float('inf') * numpy.ones((M, N))

        # グリッドグラフの1行目、1列目だけ先に処理しておく
        cost[0, 0] = self.distance(self.source[0], self.target[0])
        for i in range(1, M):
            cost[i, 0] = cost[i - 1, 0] + self.distance(self.source[i], self.target[0])
        for i in range(1, N):
            cost[0, i] = cost[0, i - 1] + self.distance(self.source[0], self.target[i])

        # 各頂点までの最短パスの長さを計算する
        for i in range(1, M):
            # 各フレームの前後self.windowフレームだけを参照する
            for j in range(max(1, i - self.window), min(N, i + self.window)):
                cost[i, j] = \
                    min(cost[i - 1, j - 1], cost[i, j - 1], cost[i - 1, j]) + \
                    self.distance(self.source[i], self.target[j])

        m, n = M - 1, N - 1
        self.path = []

        # 最短パスの経路を逆順に求めていく
        while (m, n) != (0, 0):
            self.path.append((m, n))
            m, n = min((m - 1, n), (m, n - 1), (m - 1, n - 1), \
                       key=lambda x: cost[x[0], x[1]])
            if m < 0 or n < 0:
                break

        self.path.append((0, 0))

    def align(self, data, reverse=False):
        # reverse = Trueの時は、targetのフレーム数に合わせるようにする
        if reverse:
            path = [(t[1], t[0]) for t in self.path]
            source = self.target
            target = self.source
        else:
            path = self.path
            source = self.source
            target = self.target

        path.sort(key=lambda x: (x[1], x[0]))

        shape = tuple([path[-1][1] + 1] + list(data.shape[1:]))
        alignment = numpy.ndarray(shape)

        idx = 0
        frame = 0
        candicates = []

        while idx < len(path) and frame < target.shape[0]:
            if path[idx][1] > frame:
                # 候補となっているフレームから最も類似度が高いフレームを選ぶ
                candicates.sort(key=lambda x: \
                    self.distance(source[x], target[frame]))
                alignment[frame] = data[candicates[0]]

                candicates = [path[idx][0]]
                frame += 1
            else:
                candicates.append(path[idx][0])
                idx += 1

        if frame < target.shape[0]:
            candicates.sort(key=lambda x: self.distance(source[x], target[frame]))
            alignment[frame] = data[candicates[0]]

        return alignment