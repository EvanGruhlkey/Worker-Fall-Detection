import json
from types import SimpleNamespace

from temporal.dataset import samples_from_pose_json, split_by_source, windows_from_frames


def person(track_id, x=100):
    return {'track_id':track_id,'bbox':[x,20,x+40,120],
            'keypoints':[{'x':x+20,'y':60,'confidence':.9} for _ in range(12)]}


def test_batch_json_uses_sample_index_not_original_source_frame(tmp_path):
    path=tmp_path/'pose.json'
    path.write_text(json.dumps({'metadata':{'fps':12,'source_fps':24},'frames':[
        {'sample_index':i,'source_frame':i*2,'timestamp':i/12,'people':[person(1)]}
        for i in range(40)]}))
    samples,fps=samples_from_pose_json(path,falls_seconds=[(2.6,2.8)],target_fps=12,stride=5)
    assert fps==12
    assert [(s.end_frame,s.label) for s in samples]==[(30,0),(35,1)]


def test_faller_id_handoff_keeps_fall_positive_and_filters_bystander():
    frames=[[person(1 if i<32 else 9),person(2,x=600)] for i in range(40)]
    samples=windows_from_frames(frames,12,[(32,34)],stride=5,track_ids=[1])
    assert [(s.track_id,s.end_frame,s.label) for s in samples]==[
        (1,30,0),(9,35,1)]


def test_scene_groups_never_cross_train_validation_and_test():
    samples=[SimpleNamespace(source=f'clip{i}',group='shared' if i<6 else f'group{i}',label=i%2,
                             split_key='shared' if i<6 else f'group{i}')
             for i in range(20)]
    parts=split_by_source(samples,seed=0)
    memberships=[{s.group for s in p} for p in parts]
    assert not memberships[0]&memberships[1]
    assert not memberships[0]&memberships[2]
    assert not memberships[1]&memberships[2]
    assert sum(bool([s for s in p if s.group=='shared']) for p in parts)==1
