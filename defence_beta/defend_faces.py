#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Benjamin Zhao
"""
import numpy as np
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Original definition from synthetic data tests
def sample_clipped(center, sd, dim, n, low=0.0, upp=1.0):
    data = np.random.normal(center, sd, (n, dim))
    # Repeat until clipped
    while ((len(data[data.min(axis=1) < low, :]) > 0) or
            (len(data[data.max(axis=1) >= upp, :]) > 0)):
        data = data[data.min(axis=1) >= low, :]
        data = data[data.max(axis=1) < upp, :]
        r = int(n - len(data))
        new = np.random.normal(center, sd, (r, dim))
        data = np.concatenate([data, new])
    if dim == 1 and n == 1:
        return data[0][0]
    elif n == 1:
        return data[0]
    else:
        return data


def generate_protection_noise(target_data, other_data, std_ratio):
    feat_mean = np.mean(target_data, axis=0)
    # feat_std = np.std(target_data, axis=0)

    alphas = abs(feat_mean - 0.5) + 0.5
    betas = np.array([0.5]*len(alphas))

    # add a pre-specified amount of vairance to the user's existing variance
    # gen_std = feat_std*(1+std_ratio)
    gen_data = np.random.beta(alphas, betas,
                              (len(target_data), len(feat_mean)))
    gen_data = np.abs((-1*np.round(feat_mean)) + gen_data)

    noise_other_data = np.concatenate([other_data, gen_data], axis=0)

    return target_data, noise_other_data


def split_and_combine(for_split, for_combine, combine_size):
    split_ind = np.random.choice(for_split.shape[0], combine_size,
                                 replace=False)
    unsplit_ind = np.setdiff1d(np.arange(for_split.shape[0]), split_ind)

    split_data = for_split[split_ind]
    unsplit_data = for_split[unsplit_ind]
    combined_data = np.concatenate([for_combine, split_data], axis=0)

    return unsplit_data, combined_data


if __name__ == "__main__":
    from user_faces import FaceUserPopulation
    print("Run")
    embeddings_loc = "./../data/embeddings.npy"
    label_str_loc = "./../data/label_strings.npy"
    labels_loc = "./../data/labels.npy"

    n_feat = 512

    user_faces = FaceUserPopulation(embeddings_loc, labels_loc,
                                    label_str_loc, n_feat)

    user_faces.normalize_data()
    user_faces.split_user_data(0.3)

    print(user_faces)
    for u in user_faces.users.keys():
        a, b = user_faces.get_train_sets(u, concatenate=False)
        a, c = generate_protection_noise(a, b, 0.3)

        a, d = split_and_combine(c, c, a.shape[0])

        break
