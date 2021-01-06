# %%
import pandas as pd
import numpy as np
import sys

import umap
import hdbscan
from sentence_transformers import SentenceTransformer, util


def community_detection(embeddings, threshold=0.75, min_community_size=10, init_max_size=1000):
    """
    Function for Fast Community Detection
    Finds in the embeddings all communities, i.e. embeddings that are close (closer than threshold).
    Returns only communities that are larger than min_community_size. The communities are returned
    in decreasing order. The first element in each list is the central point in the community.
    """

    # Compute cosine similarity scores
    cos_scores = util.pytorch_cos_sim(embeddings, embeddings)

    # Minimum size for a community
    top_k_values, _ = cos_scores.topk(k=min_community_size, largest=True)

    # Filter for rows >= min_threshold
    extracted_communities = []
    for i in range(len(top_k_values)):
        if top_k_values[i][-1] >= threshold:
            new_cluster = []

            # Only check top k most similar entries
            top_val_large, top_idx_large = cos_scores[i].topk(k=init_max_size, largest=True)
            top_idx_large = top_idx_large.tolist()
            top_val_large = top_val_large.tolist()

            if top_val_large[-1] < threshold:
                for idx, val in zip(top_idx_large, top_val_large):
                    if val < threshold:
                        break

                    new_cluster.append(idx)
            else:
                # Iterate over all entries (slow)
                for idx, val in enumerate(cos_scores[i].tolist()):
                    if val >= threshold:
                        new_cluster.append(idx)

            extracted_communities.append(new_cluster)

    # Largest cluster first
    extracted_communities = sorted(extracted_communities, key=lambda x: len(x), reverse=True)

    # Step 2) Remove overlapping communities
    unique_communities = []
    extracted_ids = set()

    for community in extracted_communities:
        add_cluster = True
        for idx in community:
            if idx in extracted_ids:
                add_cluster = False
                break

        if add_cluster:
            unique_communities.append(community)
            for idx in community:
                extracted_ids.add(idx)

    return unique_communities
# %%
BASE_DIR = '/home/burtenshaw/now/spans_toxic'

folds_dir_list = ['0', '1', '2', '3', '4']

for fold in folds_dir_list:

    pred_dir = os.path.join(BASE_DIR, 'predictions', fold)
    data_dir = os.path.join(BASE_DIR, 'data', fold)


model = SentenceTransformer('distilbert-base-nli-mean-tokens')
BERT_embeddings = model.encode(ENTITY_CHUNKS)



clusters = community_detection(BERT_embeddings, min_community_size=100, threshold=0.7)
# %%

#Print all cluster / communities

cluster_sentences = {}

for i, cluster in enumerate(clusters):
    print("\nCluster {}, #{} Elements ".format(i+1, len(cluster)))
    cluster_sentences[i] = ENTITY_CHUNKS[cluster[0]] 
    for sentence_id in cluster:
        df.at[sentence_id,'BERT_cos_cluster'] = i
        print("\t", ENTITY_CHUNKS[sentence_id])

# %%
cluster_sent_df = pd.DataFrame(cluster_sentences)
# %%
    # df.to_pickle('/home/burtenshaw/now/potter_kg/data/clustered_26_11_2020.bin')
cluster_sent_df.to_pickle('/home/burtenshaw/now/potter_kg/data/cluster_sent_df.bin')
# %%
