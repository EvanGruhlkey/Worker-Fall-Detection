"""Replay saved pose sequences through the trained detector, without changing videos."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

from utils.sliding_window import TrackWindows
from temporal.detector import TemporalFallDetector, handoff_lost_tracks


def main():
    import torch
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'models/fall_gru_rig.pt')
    parser.add_argument('--manifest',type=Path,default=ROOT/'data/labels/manifest.json')
    parser.add_argument('--audit',type=Path,default=ROOT/'models/fall_gru_rig.data.json')
    args=parser.parse_args()
    torch.set_num_threads(4)
    manifest=json.loads(args.manifest.read_text())
    audit=json.loads(args.audit.read_text())
    membership={clip:split for split,part in audit['splits'].items() for clip in part['clips']}
    results=[]
    started=time.perf_counter()
    for video in manifest['videos']:
        detector=TemporalFallDetector(args.checkpoint)
        windows=TrackWindows(window_size=detector.window_size)
        faller_ids=set(video.get('track_ids',[]))
        poses=Path(video['pose_json'])
        if not poses.is_absolute():
            poses=args.manifest.parent/poses
        data=json.loads(poses.read_text())
        fps=data['metadata']['fps']
        intervals=video.get('falls_seconds',[])
        matched=set()
        events=[]
        scores=[]
        for i,frame in enumerate(data['frames']):
            t=frame.get('timestamp',i/fps)
            people=frame['people']
            for old,new in handoff_lost_tracks(windows,people,i):
                if old in faller_ids:
                    faller_ids.remove(old)
                    faller_ids.add(new)
            windows.update(i,people)
            alerts=detector.update(people,t)
            scores.append({'timestamp':t,'probabilities':dict(detector.probabilities),
                           'faller_ids':sorted(faller_ids)})
            for alert in alerts:
                target=alert.track_id in faller_ids
                candidates=[j for j,(a,b) in enumerate(intervals)
                            if target and a<=t<=b+detector.window_size/fps and j not in matched]
                event={**asdict(alert),'timestamp':t,'is_target':target,'matched_fall':None}
                if candidates:
                    j=candidates[0]
                    matched.add(j)
                    event['matched_fall']=j
                    event['delay_from_start_seconds']=t-intervals[j][0]
                    event['delay_after_impact_seconds']=t-intervals[j][1]
                events.append(event)
        result={'name':video['name'],'split':membership[video['name']],
                'falls':len(intervals),'detected_falls':len(matched),'missed_falls':len(intervals)-len(matched),
                'false_alerts':sum(e['matched_fall'] is None for e in events),'events':events,
                'duration_seconds':data['metadata']['source_frames']/data['metadata']['source_fps'],
                'scores':scores}
        results.append(result)
    summary={}
    for split in ['train','val','test']:
        clips=[r for r in results if r['split']==split]
        falls=sum(r['falls'] for r in clips)
        tp=sum(r['detected_falls'] for r in clips)
        fp=sum(r['false_alerts'] for r in clips)
        delays=[e['delay_from_start_seconds'] for r in clips for e in r['events'] if e['matched_fall'] is not None]
        summary[split]={'clips':len(clips),'falls':falls,'detected_falls':tp,'missed_falls':falls-tp,
                        'false_alerts':fp,'event_precision':tp/(tp+fp) if tp+fp else 0,
                        'event_recall':tp/falls if falls else 0,
                        'mean_delay_from_start_seconds':sum(delays)/len(delays) if delays else None,
                        'normal_clips':sum(r['falls']==0 for r in clips),
                        'normal_clips_with_alerts':sum(r['falls']==0 and r['false_alerts']>0 for r in clips)}
    output=args.checkpoint.with_suffix('.events.json')
    output.write_text(json.dumps({'checkpoint':str(args.checkpoint),'threshold':detector.threshold,
        'debounce_frames':detector.gate.debounce,'match_tolerance_after_impact_seconds':detector.window_size/fps,
        'matching':'Same handoff logic as live inference; target identity plus annotated fall interval and 2.5 seconds after impact. Identity matching is automated and can be wrong in crowded scenes.',
        'evaluation_seconds':time.perf_counter()-started,'summary':summary,'clips':results},indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    print(f'Saved {output}')


if __name__=='__main__':
    main()
