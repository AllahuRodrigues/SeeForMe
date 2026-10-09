"""Create reproducible scenario-combination *specifications*, not rendered videos.

Use these rows to direct simulation expansion and manual data collection.
Never treat rows as observations, human annotations, or deployable policies.
"""
import argparse, csv, json, random
from pathlib import Path

HERE=Path(__file__).resolve().parent

def sample_catalog(n_per_case=8,seed=42):
    if n_per_case<1:
        raise ValueError('n_per_case must be positive')
    data=json.loads((HERE/'scenario_catalog.json').read_text())
    r=random.Random(seed)
    groups=['light','camera','density','geometry','speech','perception','user_goal','reaction_time','language']
    rows=[]
    for sc in data['scenarios']:
        for k in range(n_per_case):
            factors={g:r.choice(data['modifiers'][g]) for g in groups}
            # All emergency/road configurations held out as OOD. Not simulated action ground truth.
            split='ood_test' if sc['scope']=='ood_only' else 'unspecified_group_split'
            rows.append({'instance_id':f"{sc['scenario_id']}-{k:02d}", 'scenario_id':sc['scenario_id'],
                         'domain':sc['domain'],'case':sc['case'],'scope':sc['scope'],
                         'split':split,'labels':'NOT_ANNOTATED',**factors})
    return rows

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--per-case',type=int,default=8)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--out',type=Path,default=HERE/'stress_matrix.csv')
    args=ap.parse_args()
    rows=sample_catalog(args.per_case,args.seed)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    print(f'Wrote {len(rows)} SPECIFICATIONS, not videos or labeled training episodes: {args.out}')
if __name__=='__main__':main()
