#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Benjamin Zhao
"""

import pandas as pd
from user_faces import FaceUserPopulation


if __name__ == "__main__":
    print("Run")
    embeddings_loc = "./data/embeddings.npy"
    label_str_loc = "./data/label_strings.npy"
    labels_loc = "./data/labels.npy"

    n_feat = 512

    user_faces = FaceUserPopulation(embeddings_loc, labels_loc,
                                    label_str_loc, n_feat)

    print(user_faces)

    print([u_data.get_user_data() for u, u_data in user_faces.users.items() if u == 1])
    user_faces.normalize_data()
    print([u_data.get_user_data() for u, u_data in user_faces.users.items() if u == 1])

    a = pd.DataFrame(user_faces.scaler.transform(user_faces.embeddings))
    a['user'] = user_faces.labels

    a.to_csv('./data/face_data_dump.csv')
