#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import pandas as pd
SPLITS={"random":range(5),"scaffold":range(5),"repeated_stratified_perimeter":range(5),"strict_perimeter":[0]}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--data-root",default=os.environ.get("DATA_ROOT"),required=not bool(os.environ.get("DATA_ROOT")),help="Prepared split directory (or DATA_ROOT)"); ap.add_argument("--dataset-config",default=str(Path(__file__).resolve().parent / "dataset_config.csv")); a=ap.parse_args()
 root=Path(a.data_root); cfg=pd.read_csv(a.dataset_config); rows=[]
 for ds in cfg["dataset_name"].astype(str):
  for task in ["classification","regression"]:
   target="active" if task=="classification" else "log10_mic"
   for st,seeds in SPLITS.items():
    for seed in seeds:
     for part in ["train","validation","test"]:
      p=root/ds/task/st/f"seed_{seed}"/f"{part}.csv"; r={"dataset_name":ds,"task":task,"split_type":st,"seed":seed,"part":part,"path":str(p.relative_to(root)),"exists":p.exists(),"n":None,"has_smiles":False,"has_target":False,"error":""}
      if p.exists():
       try:
        df=pd.read_csv(p); cols={c.lower() for c in df.columns}; r["n"]=len(df); r["has_smiles"]=bool(cols & {"smiles","smiles_std","canonical_smiles"}); r["has_target"]=target in cols
       except Exception as e: r["error"]=type(e).__name__
      rows.append(r)
 report=pd.DataFrame(rows); report.to_csv("benchmark_data_layout_report.csv",index=False); bad=report[(~report.exists)|(~report.has_smiles)|(~report.has_target)|(report.error!="")]
 print(json.dumps({"expected_files":len(report),"ok_files":len(report)-len(bad),"bad_files":len(bad),"report":"benchmark_data_layout_report.csv"},ensure_ascii=False,indent=2))
 if len(bad): print(bad.head(50).to_string(index=False)); raise SystemExit(2)
if __name__=="__main__": main()
