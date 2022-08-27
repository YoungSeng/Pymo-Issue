import glob
import pdb

from sklearn.pipeline import Pipeline
import os
from pymo.parsers import BVHParser
from pymo.preprocessing import *


target_joints = ['Spine', 'Spine1', 'Spine2', 'Spine3', 'Neck', 'Neck1', 'Head',
                 'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
                 'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand']


def get_joint_tree(path):
    p = BVHParser()
    X = p.parse(path)

    joint_name_to_idx = {}
    for i, joint in enumerate(X.traverse()):
        joint_name_to_idx[joint] = i

    # traverse tree
    joint_links = []
    stack = [X.root_name]
    while stack:
        joint = stack.pop()
        parent = X.skeleton[joint]['parent']
        # tab = len(stack)
        # print('%s- %s (%s)'%('| '*tab, joint, parent))
        if parent:
            joint_links.append((joint_name_to_idx[parent], joint_name_to_idx[joint]))
        for c in X.skeleton[joint]['children']:
            stack.append(c)

    print(joint_name_to_idx)
    print(joint_links)


def process_bvh(gesture_filename):
    p = BVHParser()

    data_all = list()
    data_all.append(p.parse(gesture_filename))

    data_pipe = Pipeline([
        ('dwnsampl', DownSampler(tgt_fps=20, keep_all=False)),
        ('root', RootTransformer('hip_centric')),
        # ('jtsel', JointSelector(target_joints, include_root=True)),
        ('param', MocapParameterizer('position')),        # expmap, position
        ('cnst', ConstantsRemover()),
        ('np', Numpyfier())
    ])

    out_data = data_pipe.fit_transform(data_all)
    out_data = out_data[0]

    return out_data


def bvh_to_npy(bvh_path, sav_path):
    print(bvh_path)
    pos_data = process_bvh(bvh_path)
    # pos_data = np.pad(pos_data, ((0, 0), (3, 0)), 'constant', constant_values=(0, 0))
    print(pos_data.shape)
    npy_path = os.path.join(sav_path, bvh_path.split('/')[-1].replace('.bvh', '.npy'))
    np.save(npy_path, pos_data)


if __name__ == '__main__':
    '''
    cd process/
    python bvh_to_position.py
    '''
    # print joint tree information
    # bvh_file = ".../data/Trinity_Speech-Gesture_I/GENEA_Challenge_2020_data_release/Test_data/Motion/TestSeq001.bvh"
    # get_joint_tree(bvh_file)

    bvh_dir = ".../tmp/TEST/bvh/"
    save_dir = ".../tmp/TEST/npy_position/"

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # parse bvh
    files = sorted([f for f in glob.iglob(bvh_dir + '*.bvh')])

    for bvh_path in files:
        bvh_to_npy(bvh_path, save_dir)
