
import sys
import torch
sys.path.append(".")
import numpy as np
from utils.smpl import gen_canonical_pose

def save_canonical_pose(type='t-pose'):
    
    pose = gen_canonical_pose(smpl_type="smpl", canonical_type=type)
    global_orient = torch.zeros((1, 3), dtype=torch.float32)
    pose = torch.cat([global_orient, pose], dim=1)
    pose = pose.cpu().numpy()
    
    np.save(f"novel_poses/poses/{type.replace('-', '_')}_smpl.npy", pose)
    

if __name__ == "__main__":
    
    
    save_canonical_pose(type='t-pose')
    save_canonical_pose(type='da-pose')