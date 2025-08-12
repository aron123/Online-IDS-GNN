import pandas as pd
import ast
import torch
import dgl

from dataset import GraphDataset

def process_data(logger, data_csv_path, timegroup_label_csv_path, label_mapping_csv_path):
    logger.info(f'Processing data from {data_csv_path} ...')

    # Load CSV files
    df = pd.read_csv(data_csv_path)
    timegroup_labels = pd.read_csv(timegroup_label_csv_path)
    label_mapping_df = pd.read_csv(label_mapping_csv_path)

    # Create mappings
    timegroup_to_label = dict(zip(timegroup_labels['time_group'], timegroup_labels['label']))
    label_to_index = dict(zip(label_mapping_df['label'], label_mapping_df.index))

    dataset = GraphDataset()

    grouped = df.groupby('time_group')

    for i, (time_group, group_df) in enumerate(grouped):
        if i % 100 == 0:
            logger.info(f"[{i+1}/{len(grouped)}] Processing time_group {time_group} with {len(group_df)} edges ...")

        if time_group not in timegroup_to_label:
            logger.error(f"No label found for time group {time_group}. Skipping this group.")
            continue  # skip unknown groups

        node_map = {}
        node_id = 0
        edges_src, edges_dst, edge_feats = [], [], []

        for _, row in group_df.iterrows():
            src, dst = row['src'], row['dst']
            features = torch.tensor(ast.literal_eval(row['features']), dtype=torch.float32)

            for node in (src, dst):
                if node not in node_map:
                    node_map[node] = node_id
                    node_id += 1

            src_id = node_map[src]
            dst_id = node_map[dst]

            edges_src.append(src_id)
            edges_dst.append(dst_id)
            edge_feats.append(features)

        g = dgl.graph((edges_src, edges_dst), num_nodes=len(node_map))
        g.edata['feat'] = torch.stack(edge_feats)

        feat_dim = edge_feats[0].shape[0]
        g.ndata['feat'] = torch.ones((g.num_nodes(), feat_dim), dtype=torch.float32)

        label_str = timegroup_to_label[time_group]
        label_idx = label_to_index[label_str]

        dataset.add_item(g, label_idx)

    logger.info(f'Processing of {data_csv_path} is done.')

    return dataset