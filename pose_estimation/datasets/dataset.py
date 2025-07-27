# Author: Hiroyasu Akada

import os


def get_dataset(dataset_type, root, split, **kwargs):
    assert split in ["train", "test", "validation"]

    if dataset_type == "ego4view_syn_heatmap":
        from .ego4view_syn.ego4view_syn_heatmap import Ego4ViewSynHeatmapDataset
        return Ego4ViewSynHeatmapDataset(
            data_root=root,
            info_json=os.path.join(root, "{}.txt".format(split)),
            **kwargs
        )
    elif dataset_type == "ego4view_syn_heatmap_mvf":
        from .ego4view_syn.ego4view_syn_heatmap_mvf import Ego4ViewSynHeatmapMVFDataset
        return Ego4ViewSynHeatmapMVFDataset(
            data_root=root,
            info_json=os.path.join(root, "{}.txt".format(split)),
            **kwargs
        )
    elif dataset_type == "ego4view_syn_pose3d":
        from .ego4view_syn.ego4view_syn_pose3d import Ego4ViewSynPose3DDataset
        return Ego4ViewSynPose3DDataset(
            data_root=root,
            info_json=os.path.join(root, "{}.txt".format(split)),
            **kwargs
        )

    elif dataset_type == "ego4view_rw_heatmap":
        from .ego4view_rw.ego4view_rw_heatmap import Ego4ViewRWHeatmapDataset
        return Ego4ViewRWHeatmapDataset(
            data_root=root,
            info_json=os.path.join(root, "{}.txt".format(split)),
            **kwargs
        )
    elif dataset_type == "ego4view_rw_heatmap_mvf":
        from .ego4view_rw.ego4view_rw_heatmap_mvf import Ego4ViewRWHeatmapMVFDataset
        return Ego4ViewRWHeatmapMVFDataset(
            data_root=root,
            info_json=os.path.join(root, "{}.txt".format(split)),
            **kwargs
        )
    elif dataset_type == "ego4view_rw_pose3d":
        from .ego4view_rw.ego4view_rw_pose3d import Ego4ViewRWPose3DDataset
        return Ego4ViewRWPose3DDataset(
            data_root=root,
            info_json=os.path.join(root, "{}.txt".format(split)),
            **kwargs
        )

    else:
        raise NotImplementedError
