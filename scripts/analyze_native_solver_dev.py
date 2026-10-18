#!/usr/bin/env python3
"""Read-only stdlib analysis of native-mid-solver-audit cases.jsonl receipts."""
import argparse, json, math, statistics
from pathlib import Path

SOLVERS = ("euler8", "euler4", "heun2", "midpoint1")
CHANNELS = {"lr": ("left", "right"), "ms": ("mid", "side")}

def mean(xs): return statistics.mean(xs) if xs else None
def median(xs): return statistics.median(xs) if xs else None
def q(xs, p):
    if not xs: return None
    ys=sorted(xs); return ys[round((len(ys)-1)*p)]
def f(x): return None if x is None else float(f"{x:.12g}")
def score(metrics, group, key="target_missing_band"):
    chosen=[r[key] for r in metrics if r.get("channel") in CHANNELS[group]]
    power=sum(x["target_power"] for x in chosen)
    err=sum(x["absolute_error_power"] for x in chosen)
    return (err/power if power>1e-14 else None, err, power)
def read_backend(path):
    manifest=json.loads((path/"manifest.json").read_text())
    done=json.loads((path/"completion.json").read_text())
    rows=[json.loads(line) for line in (path/"cases.jsonl").read_text().splitlines() if line.strip()]
    cases=[r for r in rows if r["schema"]=="native-mid-solver-case-v1"]
    controls=[r for r in rows if r["schema"]=="native-mid-solver-control-v1"]
    return manifest,done,cases,controls

def summarize_backend(path):
    manifest,done,cases,controls=read_backend(path)
    n=manifest["count"]; seeds=list(range(manifest["seed"],manifest["seed"]+n))
    checks={"completion_status":done["status"],"case_rows":len(cases),"control_rows":len(controls),
      "expected_case_rows":n*8,"expected_control_rows":n*4,"solver_failures":done["solver_failures"],
      "all_case_status_ok":all(r["result"]["status"]=="ok" for r in cases),
      "seed_coverage_exact":sorted({r["seed"] for r in cases})==seeds,
      "case_count_exact":len(cases)==n*8,"control_count_exact":len(controls)==n*4}
    by_solver={}
    for cond in ("manufactured_mid_damage","healthy_fullband_forced_head"):
      for sol in SOLVERS:
        rr=[r for r in cases if r["condition"]==cond and r["solver"]==sol]
        by_solver[f"{cond}/{sol}"]={
          "rows":len(rr),"expected_nfe_values":sorted({r["expected_nfe"] for r in rr}),
          "actual_nfe_values":sorted({r["profile"]["nfe"] for r in rr}),
          "host_call_count_histogram":{str(k):sum(r["profile"]["predictor_batch_count"]==k for r in rr) for k in sorted({r["profile"]["predictor_batch_count"] for r in rr})},
          "cache_used_values":sorted({r["profile"]["cache_used"] for r in rr}),
          "cache_eligible_values":sorted({r["profile"]["cache_eligible"] for r in rr}),
          "max_profile_hwm_mib":f(max((r["profile"]["memory_after"]["process_high_water_kib"] or 0) for r in rr)/1024) if rr else None,
          "descriptive_wall_ms_median":f(median([r["result"]["refine_and_synthesize_wall_ms"] for r in rr]))}
    condition_summary={}
    for cond in ("manufactured_mid_damage","healthy_fullband_forced_head"):
      condition_summary[cond]={}
      for group in CHANNELS:
        condition_summary[cond][group]={}
        for sol in SOLVERS:
          rr=[r for r in cases if r["condition"]==cond and r["solver"]==sol]
          vals=[]; null_scenes=[]; total_err=total_power=0.; lsd=[]; phases=[]; coverage=[]
          lsd_defined=phase_defined=coverage_defined=phase_total=0
          per_channel={}
          for r in rr:
            v,e,p=score(r["result"]["metrics"],group); total_err+=e; total_power+=p
            if v is not None: vals.append(v)
            else: null_scenes.append({"seed":r["seed"],"family":r["family"],"target_power":f(p),"absolute_error_power":f(e)})
            for m in r["result"]["metrics"]:
              if m["channel"] not in CHANNELS[group]: continue
              b=m["target_missing_band"]
              x=b.get("target_power_weighted_whole_fourier_lsd_db")
              if x is not None: lsd.append(x); lsd_defined+=1
              x=b.get("target_power_weighted_phase_mae_radians")
              if x is not None: phases.append(x); phase_defined+=1
              phase_total+=1
              x=b.get("phase_target_power_coverage")
              if x is not None: coverage.append(x); coverage_defined+=1
              ch=per_channel.setdefault(m["channel"],[])
              if b["complex_nmse"] is not None: ch.append(b["complex_nmse"])
          condition_summary[cond][group][sol]={
            "scored_scenes":len(vals),"relative_nmse_null_scene_count":len(null_scenes),"relative_nmse_null_scenes":null_scenes,
            "per_scene_pooled_nmse_mean":f(mean(vals)),"per_scene_pooled_nmse_median":f(median(vals)),
            "energy_pooled_nmse_includes_null_scene_absolute_errors":f(total_err/total_power if total_power>1e-14 else None),
            "energy_pooled_absolute_error_power_all_scenes":f(total_err),"energy_pooled_target_power_all_scenes":f(total_power),
            "targetweighted_lsd_defined_channel_rows":lsd_defined,"targetweighted_lsd_db_per_channel_row_mean":f(mean(lsd)),"targetweighted_lsd_db_per_channel_row_median":f(median(lsd)),
            "phase_mae_defined_channel_rows":phase_defined,"phase_mae_total_channel_rows":phase_total,
            "phase_mae_radians_per_channel_row_median":f(median(phases)),"phase_mae_radians_per_channel_row_p95":f(q(phases,.95)),
            "phase_coverage_defined_channel_rows":coverage_defined,"phase_target_power_coverage_per_channel_row_median":f(median(coverage)),"phase_target_power_coverage_min":f(min(coverage) if coverage else None),
            "per_channel_nmse_mean":{k:f(mean(v)) for k,v in per_channel.items()},"per_channel_nmse_defined_counts":{k:len(v) for k,v in per_channel.items()}}
    paired={}
    for group in CHANNELS:
      paired[group]={}
      for sol in SOLVERS[1:]:
        base={r["seed"]:r for r in cases if r["condition"]=="manufactured_mid_damage" and r["solver"]=="euler8"}
        alt={r["seed"]:r for r in cases if r["condition"]=="manufactured_mid_damage" and r["solver"]==sol}
        diffs=[]; ratios=[]
        for seed,b in base.items():
          a=alt[seed]; bv,_,_=score(b["result"]["metrics"],group); av,_,_=score(a["result"]["metrics"],group)
          if bv is None or av is None: continue
          d=av-bv; rec={"seed":seed,"family":b["family"],"euler8_nmse":f(bv),"candidate_nmse":f(av),"delta":f(d)}
          diffs.append((d,rec))
          if bv>0: ratios.append((av/bv,rec))
        paired[group][sol]={"paired_scored_scenes":len(diffs),"wins_vs_euler8":sum(d<0 for d,_ in diffs),
          "losses_vs_euler8":sum(d>0 for d,_ in diffs),"mean_nmse_delta":f(mean([d for d,_ in diffs])),
          "worst_absolute_regression":max(diffs,key=lambda x:x[0])[1] if diffs else None,
          "worst_ratio_vs_euler8":({**max(ratios,key=lambda x:x[0])[1],"ratio":f(max(ratios,key=lambda x:x[0])[0])} if ratios else None)}
    paired_controls={}
    for group in CHANNELS:
      paired_controls[group]={}
      for sol in SOLVERS:
        case_by_seed={r["seed"]:r for r in cases if r["condition"]=="manufactured_mid_damage" and r["solver"]==sol}
        paired_controls[group][sol]={}
        for name in ("identity_degraded_or_healthy","dsp_harmonic_prior"):
          control_by_seed={r["seed"]:r for r in controls if r["condition"]=="manufactured_mid_damage" and r["control"]==name}
          diffs=[]; ratios=[]; nulls=[]
          for seed,r in case_by_seed.items():
            c=control_by_seed[seed]; cv,ce,cp=score(c["metrics"],group); av,ae,ap=score(r["result"]["metrics"],group)
            if cv is None or av is None:
              nulls.append({"seed":seed,"family":r["family"],"candidate_target_power":f(ap),"candidate_absolute_error_power":f(ae),"control_target_power":f(cp),"control_absolute_error_power":f(ce)})
              continue
            d=av-cv; rec={"seed":seed,"family":r["family"],"control_nmse":f(cv),"candidate_nmse":f(av),"delta":f(d)}; diffs.append((d,rec))
            if cv>0: ratios.append((av/cv,rec))
          paired_controls[group][sol][name]={"paired_scored_scenes":len(diffs),"null_scene_count":len(nulls),"null_scenes":nulls,
            "wins_vs_control":sum(d<0 for d,_ in diffs),"losses_vs_control":sum(d>0 for d,_ in diffs),
            "mean_nmse_delta":f(mean([d for d,_ in diffs])),"worst_absolute_regression":max(diffs,key=lambda x:x[0])[1] if diffs else None,
            "worst_ratio_vs_control":({**max(ratios,key=lambda x:x[0])[1],"ratio":f(max(ratios,key=lambda x:x[0])[0])} if ratios else None)}
    controls_summary={}
    for cond in ("manufactured_mid_damage","healthy_fullband_forced_head"):
      controls_summary[cond]={}
      for group in CHANNELS:
        controls_summary[cond][group]={}
        for name in ("identity_degraded_or_healthy","dsp_harmonic_prior"):
          rr=[r for r in controls if r["condition"]==cond and r["control"]==name]
          vals=[]; null_scenes=[]; lsd=[]; total_err=total_power=0.
          for r in rr:
            v,e,p=score(r["metrics"],group); total_err+=e; total_power+=p
            if v is not None: vals.append(v)
            else: null_scenes.append({"seed":r["seed"],"family":r["family"],"target_power":f(p),"absolute_error_power":f(e)})
            for m in r["metrics"]:
              if m["channel"] in CHANNELS[group]:
                x=m["target_missing_band"].get("target_power_weighted_whole_fourier_lsd_db")
                if x is not None: lsd.append(x)
          controls_summary[cond][group][name]={"rows":len(rr),"scored_scenes":len(vals),"relative_nmse_null_scene_count":len(null_scenes),"relative_nmse_null_scenes":null_scenes,
            "per_scene_pooled_nmse_mean":f(mean(vals)),"per_scene_pooled_nmse_median":f(median(vals)),
            "energy_pooled_nmse_includes_null_scene_absolute_errors":f(total_err/total_power if total_power>1e-14 else None),
            "energy_pooled_absolute_error_power_all_scenes":f(total_err),"energy_pooled_target_power_all_scenes":f(total_power),
            "targetweighted_lsd_defined_channel_rows":len(lsd),"targetweighted_lsd_db_per_channel_row_mean":f(mean(lsd))}
    trusted=[]
    healthy_rms={}
    for group,chs in CHANNELS.items():
      scene_rms=[]
      for r in cases:
        if r["condition"]!="healthy_fullband_forced_head": continue
        ms=[m for m in r["result"]["metrics"] if m["channel"] in chs]
        e=sum(m["trusted_whole_clip_fourier_vs_input"]["absolute_error_power"] for m in ms)
        # Primary-band target-to-output deviation; normalized one-sided FFT Parseval energy.
        active_e=sum(m["target_missing_band"]["absolute_error_power"] for m in ms)
        scene_rms.append((math.sqrt(active_e/len(ms)),r["seed"],r["family"],r["solver"]))
      healthy_rms[group]={"forced_head_active_band_modification_rms_median":f(median([x[0] for x in scene_rms])),
        "forced_head_active_band_modification_rms_max":({"rms":f(max(scene_rms)[0]),"seed":max(scene_rms)[1],"family":max(scene_rms)[2],"solver":max(scene_rms)[3]} if scene_rms else None)}
    for r in cases:
      for m in r["result"]["metrics"]:
        b=m["trusted_whole_clip_fourier_vs_input"]
        trusted.append((b.get("complex_nmse"),b["absolute_error_power"],r["seed"],r["solver"],r["condition"],m["channel"]))
    tv=[x for x in trusted if x[0] is not None]
    trusted_summary={"max_nmse":({"nmse":f(max(tv)[0]),"seed":max(tv)[2],"solver":max(tv)[3],"condition":max(tv)[4],"channel":max(tv)[5]} if tv else None),
      "max_absolute_error_power":({"error_power":f(max(x[1] for x in trusted)),"seed":max(trusted,key=lambda x:x[1])[2],"solver":max(trusted,key=lambda x:x[1])[3],"condition":max(trusted,key=lambda x:x[1])[4],"channel":max(trusted,key=lambda x:x[1])[5]} if trusted else None)}
    return {"backend":manifest["backend"],"manifest":{"seed":manifest["seed"],"count":n,"rate":manifest["rate"],"scene_samples":manifest["scene_samples"],"scene_seconds":manifest["scene_seconds"],"checkpoint_sha256":manifest["checkpoint_sha256"],"solver_order":manifest["solver_order"],"limitations":manifest["metrics"]["limitations"]},"receipt_checks":checks,"by_condition_solver":by_solver,"target_metrics":condition_summary,"paired_vs_euler8_damaged":paired,"paired_vs_controls_damaged":paired_controls,"controls":controls_summary,"healthy_forced_head_modification":healthy_rms,"trusted_fourier_deviation_vs_input":trusted_summary}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--run",required=True,type=Path); ap.add_argument("--out",required=True,type=Path); a=ap.parse_args()
    if a.out.exists(): raise SystemExit(f"refusing to overwrite {a.out}")
    result={"schema":"native-mid-dev-independent-analysis-v1","inputs":{},"backends":{}}
    for name in ("cpu","opencl"):
      path=a.run/name
      result["inputs"][name]={"manifest":str(path/"manifest.json"),"cases":str(path/"cases.jsonl"),"completion":str(path/"completion.json")}
      result["backends"][name]=summarize_backend(path)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"out":str(a.out),"cpu_checks":result["backends"]["cpu"]["receipt_checks"],"opencl_checks":result["backends"]["opencl"]["receipt_checks"],"cpu_paired":result["backends"]["cpu"]["paired_vs_euler8_damaged"],"cpu_controls":result["backends"]["cpu"]["controls"]["manufactured_mid_damage"],"cpu_trusted":result["backends"]["cpu"]["trusted_fourier_deviation_vs_input"]},indent=2))
if __name__=="__main__": main()
