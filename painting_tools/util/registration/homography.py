import os, os.path as osp

import cv2 as cv
from kornia.feature import KeyNetHardNet
import numpy as np
import torch


def batch_sift_homography(path_list, rescale_factor=1, save_dir=None, save_filename=None, padding=0, verbose=False):
    im1 = cv.imread(path_list[0], cv.IMREAD_GRAYSCALE)
    if padding > 0:
        im1 = cv.copyMakeBorder(im1, *[padding] * 4, cv.BORDER_CONSTANT)
    if rescale_factor < 1:
        h, w = [int(sh * rescale_factor) for sh in im1.shape]
        im1 = cv.resize(im1, (w, h))

    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(im1, None)

    for i in range(1, len(path_list)):
        if verbose:
            print(i)
        im2 = cv.imread(path_list[i], cv.IMREAD_COLOR)
        im2_gray = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)
        if rescale_factor < 1:
            h, w = [int(sh * rescale_factor) for sh in im2_gray.shape]
            im2_gray = cv.resize(im2_gray, (w, h))
        M = sift_homography(im1, im2_gray, (kp1, des1))
        if M is None: continue
        im2_warped = cv.warpPerspective(im2, np.linalg.inv(M), (im2.shape[1], im2.shape[0]))

        filename = osp.split(path_list[i])[-1]
        if save_filename is not None:
            name, extension = filename.split('.')
            filename = '{}_{:03n}.{}'.format(save_filename, int(name.split('_')[-1]), extension)

        save_dir = osp.dirname(path_list[i]) if save_dir is None else save_dir
        if not osp.exists(save_dir):
            os.mkdir(save_dir)
        save_path = osp.join(save_dir, filename)
        cv.imwrite(save_path, im2_warped)


def sift_homography(im1, im2, threshold=0.2, min_match_count=2, kp_des1=None):
    sift = cv.SIFT_create()

    # Compute keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(im1, None) if kp_des1 is None else kp_des1
    kp2, des2 = sift.detectAndCompute(im2, None)

    if (len(kp1) < 2 or len(kp2) < 2):
        return np.array([
            [1., 0., 0.],
            [0., 1., 0.]
        ])

    # Find matching keypoints
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Filter out matches to only keep those matches where
    # the closest match is closer than threshold times the distance of the second closest match.
    distances_and_idx = np.array([[n1.distance, n2.distance, n1.queryIdx, n1.trainIdx] for n1, n2 in matches])
    good_matches = distances_and_idx[distances_and_idx[..., 0] < threshold * distances_and_idx[..., 1], 2:].astype(np.int32)

    # Compute translation between keypoints as difference between mean (minimizer of least squares energy) 
    # Also correct for the translation that patch1 has undergone. 
    if good_matches.shape[0] < min_match_count:
        return np.array([
            [1., 0., 0.],
            [0., 1., 0.]
        ])

    # Get the locations of the best matching keypoints
    matched_kp1 = np.array([kp1[idx].pt for idx in good_matches[..., 0]])
    matched_kp2 = np.array([kp2[idx].pt for idx in good_matches[..., 1]])

    M, _ = cv.findHomography(matched_kp2, matched_kp1, cv.RANSAC, 5.0)
   
    return M


def keynet_hardnet_homography(im1, im2, threshold=0.2, min_match_count=2, num_features=8000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    keynet_hardnet = KeyNetHardNet(num_features=num_features, upright=True, device=device)

    if im1.dtype == np.uint8:
        im1 = im1.astype(np.float32) / 255.
    if im2.dtype == np.uint8:
        im2 = im2.astype(np.float32) / 255.
        
    im1 = torch.from_numpy(im1).to(device)
    im2 = torch.from_numpy(im2).to(device)

    # Compute keypoints and descriptors
    with torch.no_grad():
        kp1, _, des1 = keynet_hardnet(im1[None, None])
        kp2, _, des2 = keynet_hardnet(im2[None, None])
    kp1 = kp1[0].cpu().numpy()
    des1 = des1[0].cpu().numpy()
    kp2 = kp2[0].cpu().numpy()
    des2 = des2[0].cpu().numpy()

    if (len(kp1) < 2 or len(kp2) < 2):
        print('Not enough keypoints')
        return np.array([
            [1., 0., 0.],
            [0., 1., 0.]
        ])

    # Find matching keypoints
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Filter out matches to only keep those matches where
    # the closest match is closer than threshold times the distance of the second closest match.
    distances_and_idx = np.array([[n1.distance, n2.distance, n1.queryIdx, n1.trainIdx] for n1, n2 in matches])
    good_matches = distances_and_idx[distances_and_idx[..., 0] < threshold * distances_and_idx[..., 1], 2:].astype(np.int32)

    # Compute translation between keypoints as difference between mean (minimizer of least squares energy) 
    # Also correct for the translation that patch1 has undergone. 
    if good_matches.shape[0] < min_match_count:
        return np.array([
            [1., 0., 0.],
            [0., 1., 0.]
        ])

    # Get the locations of the best matching keypoints
    matched_kp1 = np.array([kp1[idx, :, 2] for idx in good_matches[..., 0]])
    matched_kp2 = np.array([kp2[idx, :, 2] for idx in good_matches[..., 1]])

    M, _ = cv.findHomography(matched_kp2, matched_kp1, cv.RANSAC, 5.0)
   
    return M
