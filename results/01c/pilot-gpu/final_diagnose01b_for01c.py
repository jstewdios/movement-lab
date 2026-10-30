import json
from pathlib import Path
import numpy as np
root=Path('results/01b/departures');c=dict(np.load(root/'baseline/zero/trial.npz'));report={}
for name,seq in [('flex','flex_+1'),('head-positive','head_+1'),('head-negative','head_-1')]:
 a=dict(np.load(root/name/seq/'trial.npz'));m=json.loads((root/name/seq/'metadata.json').read_text());t=a['time_s'];out={}
 for lo,hi in [(2.8,3),(3.6,4.1),(4.5,5),(5.5,6.01)]:
  w=(t>=lo)&(t<hi);j=int(np.argmax(abs((a['joint_position_rad'][w]-c['joint_position_rad'][w]).mean(0))))
  out[f'{lo}-{hi}']=dict(joint=m['joint_names'][j],position_delta=float((a['joint_position_rad'][w,j]-c['joint_position_rad'][w,j]).mean()),actor_delta=float((a['actor_action'][w,j]-c['actor_action'][w,j]).mean()),velocity_mean=float(a['joint_velocity_rad_s'][w,j].mean()),com_delta_m=(a['com_position_w_m'][w]-c['com_position_w_m'][w]).mean(0).tolist())
 report[name]=out
Path('results/01c/evidence01b.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
