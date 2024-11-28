import torch
import numpy as np

def computeNN(desc_ii, desc_jj):
    desc_ii, desc_jj = torch.from_numpy(desc_ii).cuda(), torch.from_numpy(desc_jj).cuda()
    d1 = (desc_ii**2).sum(1)
    d2 = (desc_jj**2).sum(1)
    distmat = (d1.unsqueeze(1) + d2.unsqueeze(0) - 2*torch.matmul(desc_ii, desc_jj.transpose(0,1))).sqrt()
    distVals, nnIdx1 = torch.topk(distmat, k=2, dim=1, largest=False)
    nnIdx1 = nnIdx1[:,0]
    _, nnIdx2 = torch.topk(distmat, k=1, dim=0, largest=False)
    nnIdx2= nnIdx2.squeeze()
    mutual_nearest = (nnIdx2[nnIdx1] == torch.arange(nnIdx1.shape[0]).cuda()).cpu().numpy()
    ratio_test = (distVals[:,0] / distVals[:,1].clamp(min=1e-10)).cpu().numpy()
    idx_sort = [np.arange(nnIdx1.shape[0]), nnIdx1.cpu().numpy()]
    return idx_sort, ratio_test, mutual_nearest





    





# def sub_desc_match(desc1, desc2, desc_size):
#     splitted_desc1 = np.split(desc1, desc_size,1)
#     splitted_desc2 = np.split(desc2, desc_size,1)
#     hist = np.zeros(shape=(len(desc1), len(desc2)))
#     for d1, d2 in zip (splitted_desc1, splitted_desc2):
#         idx_sort, ratio_test, mutual_nearest = computeNN(d1, d2)
#         hist[idx_sort[0][mutual_nearest], idx_sort[1][mutual_nearest]] +=1
#     return hist
    
# def multi_sub_desc_match(desc1, desc2, desc_size, min_match):
#     hists = []
#     for d1, d2 in zip(desc1, desc2):
#         hist = sub_desc_match(d1, d2, desc_size)
#         hists.append(hist)
#     final_matches = hists[0]>min_match
#     if len(hists) >1: 
#         for h in hists[1:]:
#             final_matches = np.logical_and(final_matches, h>min_match)
            
#     final_matches = np.argwhere(final_matches)
#     idx_sort = np.c_[np.arange(len(d1)), np.random.choice(np.arange(len(d2)), size=len(d1), replace=True)]
#     idx_sort[final_matches[:, 0], 1] = final_matches[:, 1]
#     idx_sort = [idx_sort[:, 0], idx_sort[:, 1]]
#     mutual_matches =  np.zeros(len(idx_sort[0]), np.bool_)
#     mutual_matches[final_matches[:, 0]]=True
#     ratio_test = np.logical_not(mutual_matches).astype('float32')
#     return idx_sort, ratio_test, mutual_matches
    





def sub_desc_match(desc1, desc2, desc_size):
    splitted_desc1 = np.split(desc1, desc_size,1)
    splitted_desc2 = np.split(desc2, desc_size,1)
    hist = np.zeros(shape=(len(desc1), len(desc2)))
    for d1, d2 in zip (splitted_desc1, splitted_desc2):
        idx_sort, ratio_test, mutual_nearest = computeNN(d1, d2)
        hist[idx_sort[0][mutual_nearest], idx_sort[1][mutual_nearest]] +=1
    return hist
    
def multi_sub_desc_match(desc1, desc2, desc_size, min_match):
    hists = []
    for d1, d2 in zip(desc1, desc2):
        hist = sub_desc_match(d1, d2, desc_size)
        hists.append(hist)
    final_matches = hists[0]>min_match
    if len(hists) >1: 
        for h in hists[1:]:
            final_matches = np.logical_and(final_matches, h>min_match)
            
    final_matches = np.argwhere(final_matches)
    # idx_sort = np.c_[np.arange(len(d1)), np.random.choice(np.arange(len(d2)), size=len(d1), replace=True)]
    # idx_sort[final_matches[:, 0], 1] = final_matches[:, 1]
    # idx_sort = [idx_sort[:, 0], idx_sort[:, 1]]
    idx_sort = [final_matches[:, 0], final_matches[:, 1]]
    # mutual_matches =  np.zeros(len(idx_sort[0]), np.bool_)
    # mutual_matches[final_matches[:, 0]]=True
    # ratio_test = mutual_matches.astype('float32')

    return idx_sort, None, None
    

    
    
    
    
    
    