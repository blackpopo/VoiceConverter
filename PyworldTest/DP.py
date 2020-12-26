import  sys
import numpy
import scipy

def dtw(source, target, window, distance):
    M, N = len(source), len(target)
    cost = sys.maxsize * numpy.ones((M, N))

    # グリッドグラフの1行目、1列目だけ先に処理しておく
    cost[0, 0] = distance(source[0], target[0])
    for i in range(1, M):
        cost[i, 0] = cost[i - 1, 0] + distance(source[i], target[0])
    for i in range(1, N):
        cost[0, i] = cost[0, i - 1] + distance(source[0], target[i])

    # 各頂点までの最短パスの長さを計算する
    for i in range(1, M):
        # 各フレームの前後windowフレームだけを参照する
        for j in range(max(1, i - window), min(N, i + window)):
            cost[i, j] = \
                min(cost[i - 1, j - 1], cost[i, j - 1], cost[i - 1, j]) + \
                distance(source[i], target[j])

    m, n = M - 1, N - 1
    path = []

    # 最短パスの経路を逆順に求めていく
    while (m, n) != (0, 0):
        path.append((m, n))
        m, n = min((m - 1, n), (m, n - 1), (m - 1, n - 1), \
                   key=lambda x: cost[x[0], x[1]])
        if m < 0 or n < 0:
            break

    path.append((0, 0))
    return path

def cosine(A, B):
    return scipy.dot(A, B.transpose())

def euclidean(A, B):
    return scipy.linalg.norm(A -B)

#dataは並び変える法
def align(source, target, window, distance, data, reverse=False):
    # reverse = Trueの時は、targetのフレーム数に合わせるようにする
    if distance=='cosline':
        distance = cosine
    else:
        distance = euclidean
    path = dtw(source, target, window, distance)
    print('finish')
    if reverse:
        path = [(t[1], t[0]) for t in path]
        source = target
        target = source
    else:
        path = path
        source = source
        target = target

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
                distance(source[x], target[frame]))
            alignment[frame] = data[candicates[0]]

            candicates = [path[idx][0]]
            frame += 1
        else:
            candicates.append(path[idx][0])
            idx += 1

    if frame < target.shape[0]:
        candicates.sort(key=lambda x: distance(source[x], target[frame]))
        alignment[frame] = data[candicates[0]]

    return alignment

