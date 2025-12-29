from numpy import array, zeros, full, argmin, inf, ndim
import numpy as np
from math import isinf
from tqdm import tqdm
from mGPT.utils.human_models import rigid_align


"""
Dynamic time warping (DTW) is used as a similarity measured between temporal sequences. 
Original DTW code found at https://github.com/pierre-rouanet/dtw
"""

# Apply DTW
def dtw(x, y, dist, warp=1, w=inf, s=1.0):
    """
    Computes Dynamic Time Warping (DTW) of two sequences.

    :param array x: N1*M array
    :param array y: N2*M array
    :param func dist: distance used as cost measure
    :param int warp: how many shifts are computed.
    :param int w: window size limiting the maximal distance between indices of matched entries |i,j|.
    :param float s: weight applied on off-diagonal moves of the path. As s gets larger, the warping path is increasingly biased towards the diagonal
    Returns the minimum distance, the cost matrix, the accumulated cost matrix, and the wrap path.
    """
    assert len(x)
    assert len(y)
    assert isinf(w) or (w >= abs(len(x) - len(y)))
    assert s > 0
    # print(x.shape, y.shape)
    r, c = len(x), len(y)
    if not isinf(w):
        D0 = full((r + 1, c + 1), inf)
        for i in range(1, r + 1):
            D0[i, max(1, i - w):min(c + 1, i + w + 1)] = 0
        D0[0, 0] = 0
    else:
        D0 = zeros((r + 1, c + 1))
        D0[0, 1:] = inf
        D0[1:, 0] = inf
    D1 = D0[1:, 1:]  # view
    for i in range(r):
        for j in range(c):
            if (isinf(w) or (max(0, i - w) <= j <= min(c, i + w))):
                D1[i, j] = dist(x[i], y[j])
    C = D1.copy()
    jrange = range(c)
    for i in range(r):
        if not isinf(w):
            jrange = range(max(0, i - w), min(c, i + w + 1))
        for j in jrange:
            min_list = [D0[i, j]]
            for k in range(1, warp + 1):
                i_k = min(i + k, r)
                j_k = min(j + k, c)
                min_list += [D0[i_k, j] * s, D0[i, j_k] * s]
            D1[i, j] += min(min_list)
    if len(x) == 1:
        path = zeros(len(y)), range(len(y))
    elif len(y) == 1:
        path = range(len(x)), zeros(len(x))
    else:
        path = _traceback(D0)
    return D1[-1, -1], C, D1, path

def _traceback(D):
    i, j = array(D.shape) - 2
    p, q = [i], [j]
    while (i > 0) or (j > 0):
        tb = argmin((D[i, j], D[i, j + 1], D[i + 1, j]))
        if tb == 0:
            i -= 1
            j -= 1
        elif tb == 1:
            i -= 1
        else:  # (tb == 2):
            j -= 1
        p.insert(0, i)
        q.insert(0, j)
        # print(i,j)
    return array(p), array(q)


def l2_dist_align(x, y, wanted=None, align_idx=None):
    # joints -> align -> L2 dist
    # x,y [N,3]
    # print(x.shape, max(wanted))
    if align_idx is None:
        x = rigid_align(x, y)
    else:
        x = x - x[align_idx:align_idx+1] + y[align_idx:align_idx+1]
    if wanted is not None:
        x = x[wanted]
        y = y[wanted]
    dist = np.mean(np.sqrt(((x-y)**2).sum(axis=1)))
    # print(dist)
    return dist


def l2_dist(x, y, wanted=None):
    # x,y [N,3]
    # print(x.shape, max(wanted))
    if wanted is not None:
        x = x[wanted]
        y = y[wanted]
    dist = np.mean(np.sqrt(((x-y)**2).sum(axis=1)))
    # print(dist)
    return dist


def l1_dist(x, y, wanted=None):
    # x,y [N,3]
    # print(x.shape, max(wanted))
    if wanted is not None:
        x = x[wanted]
        y = y[wanted]
    dist = np.mean(np.abs(((x-y)).sum(axis=0)))
    # print(dist)
    return dist


# if __name__ == '__main__':
#     w = inf
#     s = 1.0
#     # if 1:  # 1-D numeric
#     #     from sklearn.metrics.pairwise import manhattan_distances
#     #     x = [0, 0, 1, 1, 2, 4, 2, 1, 2, 0]
#     #     y = [1, 1, 1, 2, 2, 2, 2, 3, 2, 0]
#     #     dist_fun = manhattan_distances
#     #     w = 1
#     #     # s = 1.2
#     # elif 0:  # 2-D numeric
#     #     from sklearn.metrics.pairwise import euclidean_distances
#     #     x = [[0, 0], [0, 1], [1, 1], [1, 2], [2, 2], [4, 3], [2, 3], [1, 1], [2, 2], [0, 1]]
#     #     y = [[1, 0], [1, 1], [1, 1], [2, 1], [4, 3], [4, 3], [2, 3], [3, 1], [1, 2], [1, 0]]
#     #     dist_fun = euclidean_distances
    
#     wanted = list(range(11)) + list(range(91, 133)) + list(range(71,91,2)) + list(range(40, 71, 3))
#     wanted = list(range(91, 133))
#     wanted = np.array(wanted)
#     dist_fun = partial(cust_dist_fun_temp, wanted=wanted)
#     print(len(wanted))

#     #------------------------------------------------2d kp loss--------------------------------------------------
#     with open('../../data/phoenix_2014t/keypoints_hrnet_dark_coco_wholebody.pkl', 'rb') as f:
#         gt_kps = pickle.load(f)
#     with open('../../data/phoenix_syn/keypoints_2d_smplx_ori.pkl', 'rb') as f:
#         smplx_kps = pickle.load(f)
#     with open('../../data/phoenix_syn/keypoints_2d_mesh.pkl', 'rb') as f:
#         ours_kps = pickle.load(f)
#     with open('../../data/phoenix_syn/keypoints_2d_osx.pkl', 'rb') as f:
#         osx_kps = pickle.load(f)
#     with open('../../data/phoenix_syn/keypoints_2d_smplerx.pkl', 'rb') as f:
#         smplerx_kps = pickle.load(f)
#     with gzip.open('../../data/phoenix_2014t/phoenix14t.dev', 'rb') as f:
#         dev = pickle.load(f)
#     with gzip.open('../../data/phoenix_2014t/phoenix14t.test', 'rb') as f:
#         test = pickle.load(f)
    
#     dist_smplx_lst, dist_ours_lst, dist_osx_lst, dist_smplerx_lst = [], [], [], []
#     for item in tqdm(dev):
#         name = item['name']
#         gt = gt_kps[name]['keypoints']
#         try:
#             smplx = smplx_kps[name]['keypoints']
#             ours = ours_kps[name]['keypoints']
#             osx = osx_kps[name]['keypoints']
#             smplerx = smplerx_kps[name]['keypoints']
#         except:
#             continue
        
#         if np.sum(np.isnan(gt)) > 0 or np.sum(np.isinf(gt)) > 0 or \
#             np.sum(np.isnan(smplx)) > 0 or np.sum(np.isnan(ours)) > 0 or np.sum(np.isnan(osx)) > 0 or np.sum(np.isnan(smplerx)) > 0:
#             continue
#         if smplx.shape[0] != gt.shape[0] or ours.shape[0] != gt.shape[0] or osx.shape[0] != gt.shape[0] or smplerx.shape[0] != gt.shape[0]:
#             continue

#         smplx_loss = dist_fun(gt, smplx)
#         ours_loss = dist_fun(gt, ours)
#         osx_loss = dist_fun(gt, osx)
#         smplerx_loss = dist_fun(gt, smplerx)
#         if ours_loss < smplerx_loss and ours_loss < osx_loss and ours_loss < smplx_loss and osx_loss < smplx_loss and smplerx_loss < smplx_loss:
#             dist_smplx_lst.append(smplx_loss)
#             dist_ours_lst.append(ours_loss)
#             dist_osx_lst.append(osx_loss)
#             dist_smplerx_lst.append(smplerx_loss)

#     dist_smplx = np.ma.masked_invalid(np.array(dist_smplx_lst)).mean()
#     dist_ours = np.ma.masked_invalid(np.array(dist_ours_lst)).mean()
#     dist_osx = np.ma.masked_invalid(np.array(dist_osx_lst)).mean()
#     dist_smplerx = np.ma.masked_invalid(np.array(dist_smplerx_lst)).mean()

#     print('smplx: ', dist_smplx, 'smplerx: ', dist_smplerx, 'osx: ', dist_osx, 'ours: ', dist_ours)



    #------------------------------------------------dtw--------------------------------------------------
    # with open('../../data/phoenix_2014t/keypoints_hrnet_dark_coco_wholebody.pkl', 'rb') as f:
    #     kps_hrnet = pickle.load(f)
    # with open('../../data/phoenix_syn/keypoints_2d_smplx_ori.pkl', 'rb') as f:
    #     kps_smplx = pickle.load(f)
    # with open('../../data/phoenix_syn/keypoints_2d_mesh.pkl', 'rb') as f:
    #     kps_ours = pickle.load(f)
    # with open('../../data/phoenix_syn/keypoints_2d_osx_cam_000000_099999.pkl', 'rb') as f:
    #     kps_osx = pickle.load(f)
    # with open('../../data/phoenix_syn/keypoints_2d_smplerx_cam_000000_099999.pkl', 'rb') as f:
    #     kps_smplerx = pickle.load(f)

    # with gzip.open('../../data/phoenix_2014t/phoenix14t.dev', 'rb') as f:
    #     dev = pickle.load(f)
    # with gzip.open('../../data/phoenix_2014t/phoenix14t.test', 'rb') as f:
    #     test = pickle.load(f)

    # dist_smplx_lst, dist_ours_lst, dist_osx_lst, dist_smplerx_lst = [], [], [], []
    # num_nan = 0
    # for item in tqdm(dev):
    #     name = item['name']
    #     if name in kps_smplx and name in kps_ours:
    #         hrnet = kps_hrnet[name]['keypoints']
    #         T = hrnet.shape[0]
    #         smplx = kps_smplx[name]['keypoints']    
    #         ours = kps_ours[name]['keypoints']
    #         osx = kps_osx[name]['keypoints']
    #         smplerx = kps_smplerx[name]['keypoints']

    #         if np.sum(np.isinf(smplx))>0 or np.sum(np.isinf(ours))>0 or np.sum(np.isinf(osx))>0 or np.sum(np.isinf(smplerx))>0:
    #             num_nan += 1
    #             continue

    #         smplx = torch.tensor(smplx[:,wanted,:-1]).reshape(smplx.shape[0], -1).float()
    #         x = smplx[:-1, ...]
    #         y = smplx[1:, ...]
    #         sim = F.cosine_similarity(x, y)
    #         dist_smplx_lst.extend(sim.tolist())

    #         ours = torch.tensor(ours[:,wanted,:-1]).reshape(ours.shape[0], -1).float()
    #         x = ours[:-1, ...]
    #         y = ours[1:, ...]
    #         sim = F.cosine_similarity(x, y)
    #         dist_ours_lst.extend(sim.tolist())

    #         osx = torch.tensor(osx[:,wanted,:-1]).reshape(osx.shape[0], -1).float()
    #         x = osx[:-1, ...]
    #         y = osx[1:, ...]
    #         sim = F.cosine_similarity(x, y)
    #         dist_osx_lst.extend(sim.tolist())

    #         smplerx = torch.tensor(smplerx[:,wanted,:-1]).reshape(smplerx.shape[0], -1).float()
    #         x = smplerx[:-1, ...]
    #         y = smplerx[1:, ...]
    #         sim = F.cosine_similarity(x, y)
    #         dist_smplerx_lst.extend(sim.tolist())

    #         # dist_smplx, _, _, _ = dtw(hrnet, smplx, dist_fun, w=w, s=s)
    #         # dist_smplx /= T
    #         # dist_ours, _, _, _ = dtw(hrnet, ours, dist_fun, w=w, s=s)
    #         # dist_ours /= T
    #         # dist_osx, _, _, _ = dtw(hrnet, osx, dist_fun, w=w, s=s)
    #         # dist_osx /= T
    #         # dist_smplerx, _, _, _ = dtw(hrnet, smplerx, dist_fun, w=w, s=s)
    #         # dist_smplerx /= T

    #         # if dist_ours < dist_smplerx and dist_ours < dist_osx and dist_ours < dist_smplx and dist_osx < dist_smplx and dist_smplerx < dist_smplx:
    #         #     dist_smplx_lst.append(dist_smplx)
    #         #     dist_ours_lst.append(dist_ours)
    #         #     dist_osx_lst.append(dist_osx)
    #         #     dist_smplerx_lst.append(dist_smplerx)
    #         #     print('smplx: ', dist_smplx, 'smplerx: ', dist_smplerx, 'osx: ', dist_osx, 'ours: ', dist_ours)

    # dist_smplx = np.ma.masked_invalid(np.array(dist_smplx_lst)).mean()
    # dist_ours = np.ma.masked_invalid(np.array(dist_ours_lst)).mean()
    # dist_osx = np.ma.masked_invalid(np.array(dist_osx_lst)).mean()
    # dist_smplerx = np.ma.masked_invalid(np.array(dist_smplerx_lst)).mean()

    # print('smplx: ', dist_smplx, 'smplerx: ', dist_smplerx, 'osx: ', dist_osx, 'ours: ', dist_ours)
    # print(num_nan)


    # dist, cost, acc, path = dtw(x, y, dist_fun, w=w, s=s)

    # Vizualize
    # from matplotlib import pyplot as plt
    # plt.imshow(cost.T, origin='lower', cmap=plt.cm.Reds, interpolation='nearest')
    # plt.plot(path[0], path[1], '-o')  # relation
    # plt.xticks(range(len(x)), x)
    # plt.yticks(range(len(y)), y)
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.axis('tight')
    # if isinf(w):
    #     plt.title('Minimum distance: {}, slope weight: {}'.format(dist, s))
    # else:
    #     plt.title('Minimum distance: {}, window widht: {}, slope weight: {}'.format(dist, w, s))
    # plt.show()

if __name__ == '__main__':
    def func(face, hands, overall):
        return (overall*74 - face*21 - hands*42) / 11
    def func1(body, face, hands):
        return (body*11 + face*21 + hands*42) / 74
    
    # print(func(19.85, 34.62, 31.56))
    # print(func(21.72, 31.88, 29.50))
    # print(func(19.67, 29.05, 26.87))
    # print(func(19.33, 23.98, 22.09))
    print(func1(42.23, 19.85, 34.62))
    print(func1(35.27, 21.72, 31.88))
    print(func1(32.29, 19.67, 29.05))
    print(func1(20.14, 19.33, 23.98))