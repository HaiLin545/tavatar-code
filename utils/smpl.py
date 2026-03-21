import torch


def gen_canonical_pose(smpl_type="smpl", canonical_type="t-pose"):

    if smpl_type == "smpl":
        body_pose = torch.zeros((1, 69), dtype=torch.float32)  # t-pose
        if canonical_type == "t-pose":
            pass
        elif canonical_type == "da-pose":
            body_pose[:, 2] = torch.pi / 6
            body_pose[:, 5] = -torch.pi / 6
        else:
            raise ValueError("Unknown canonical pose type: {}".format(canonical_type))
    elif smpl_type == "smplx":
        body_pose = torch.zeros((1, 63), dtype=torch.float32)
        if canonical_type == "t-pose":
            pass
        elif canonical_type == "da-pose":
            body_pose[:, 2] = torch.pi / 6
            body_pose[:, 5] = -torch.pi / 6
        else:
            raise ValueError("Unknown canonical pose type: {}".format(canonical_type))

    else:
        raise ValueError("Unknown SMPL type: {}".format(smpl_type))

    return body_pose
