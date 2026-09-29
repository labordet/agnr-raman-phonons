# SPDX-License-Identifier: CC-BY-4.0
"""Verify published quantities from retained fit tables, without refitting."""
from pathlib import Path
import os
import argparse,json,math,os
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from paper_reproduction.output_paths import prepare_output_directory as _output_directory
ROOT=Path(os.environ["RAMAN_DATA_ROOT"]).resolve()

def prepare_output_directory(path):
    return _output_directory(ROOT, path)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,default=Path('outputs/paper_reproduction/reported_results'))
    args=ap.parse_args();out=prepare_output_directory(args.output)
    df=pd.read_csv(ROOT/'data/derived/peak_parameters.csv')
    fits=pd.concat([pd.read_csv(p) for p in (ROOT/'data/derived/fits').glob('*/*_long_results.csv')],ignore_index=True)
    joined=df.merge(fits,on=['family','sequence','temperature_K','peak_id'],suffixes=('_published','_fit'),validate='one_to_one')
    checks=[]
    def record(name,value,reported=None,decimals=None,unit='',location=''):
        passed=None if reported is None else abs(float(value)-reported)<=0.5*10**(-decimals)+1e-12
        checks.append({'quantity':name,'computed_value':float(value),'reported_value':reported,'display_decimals':decimals,'unit':unit,'manuscript_location':location,'matches_reported_precision':passed})
    for col in ['height','position_cm-1','width_fwhm_cm-1']:
        err=float(np.max(np.abs(joined[col+'_published']-joined[col+'_fit'])))
        if err>1e-10:raise ValueError(f'Accepted fit table mismatch in {col}')
        record('accepted_'+col+'_max_difference',err,0,8)
    for pub,src in [('position_total_std_visual','position_std_cm-1'),('width_total_std_visual','width_fwhm_std_cm-1')]:
        err=float(np.max(np.abs(joined[pub]-joined[src])))
        if err>1e-10:raise ValueError('Bootstrap uncertainty mismatch')
        record(pub+'_max_difference',err,0,8)
    if not (joined.bootstrap_success==100).all():raise ValueError('Not every accepted row records 100 successful bootstraps')
    record('accepted_spectra',len(df[['family','sequence','temperature_K']].drop_duplicates()),unit='spectra')
    record('accepted_mode_rows',len(df),unit='rows')
    ii=fits[fits.family=='Aligned_Au_8A'];v=fits[(fits.family=='Aligned_RO_8A')&(fits.sequence=='Spikes_Removed_DOWN_1')]
    a=ii[ii.sequence=='Spikes_Removed_DOWN_1']
    for mode,reported,unc,pct in [('D',3.73,1.29,22.8),('G',10.50,.51,78.0),('RBLM',-2.81,2.32,None)]:
        aa=a[(a.peak_id==mode)&(a.temperature_K==100)].iloc[0]
        vv=v[(v.peak_id==mode)&(v.temperature_K==100)].iloc[0]
        delta=vv['width_fwhm_cm-1']-aa['width_fwhm_cm-1']
        u=math.hypot(vv['width_fwhm_std_cm-1'],aa['width_fwhm_std_cm-1'])
        record('S14_'+mode+'_100K_FWHM_difference',delta,reported,2,'cm^-1','SI p29-30')
        record('S14_'+mode+'_100K_difference_uncertainty',u,unc,2,'cm^-1','SI p29-30')
        if pct is not None:record('S14_'+mode+'_100K_percent_increase',100*delta/aa['width_fwhm_cm-1'],pct,1,'%','SI p29')
    shared=a.merge(v,on=['temperature_K','peak_id'],suffixes=('_II','_V'))
    shared.to_csv(out/'configuration_II_V_matched_temperatures.csv',index=False)
    for mode,meanII,meanV,pct in [('D',16.28,19.69,20.9),('G',13.29,24.19,82.1)]:
        s=shared[shared.peak_id==mode]
        record('S14_'+mode+'_shared_temperatures',len(s),23,0,'temperatures','SI p29')
        left=s['width_fwhm_cm-1_II'].mean();right=s['width_fwhm_cm-1_V'].mean()
        record('S14_'+mode+'_mean_FWHM_II',left,meanII,2,'cm^-1','SI p29')
        record('S14_'+mode+'_mean_FWHM_V',right,meanV,2,'cm^-1','SI p29')
        record('S14_'+mode+'_mean_percent_increase',100*(right/left-1),pct,1,'%','SI p29')
        record('S14_'+mode+'_broader_V_all_temperatures',bool((s['width_fwhm_cm-1_V']>s['width_fwhm_cm-1_II']).all()),1,0)
    for name,d,centre_n,width_n in [('II',ii,0,0),('V',v,24,6)]:
        c=d[d.peak_id=='CH_MID']
        centre=np.isclose(c['position_cm-1'],c.x_bound_min,rtol=0,atol=1e-5)|np.isclose(c['position_cm-1'],c.x_bound_max,rtol=0,atol=1e-5)
        width=np.isclose(c['width_fwhm_cm-1'],c.width_bound_min,rtol=0,atol=1e-5)|np.isclose(c['width_fwhm_cm-1'],c.width_bound_max,rtol=0,atol=1e-5)
        record('S14_'+name+'_CH_MID_centre_bound_hits',centre.sum(),centre_n,0,'spectra','SI p29')
        record('S14_'+name+'_CH_MID_width_bound_hits',width.sum(),width_n,0,'spectra','SI p29')
    for mode,vals in [('D',[(80,16.42),(190,15.94),(270,17.59)]),('G',[(80,13.79),(160,12.66),(250,13.81)])]:
        for temp,val in vals:
            row=ii[(ii.sequence=='Spikes_Removed_UP_1')&(ii.peak_id==mode)&(ii.temperature_K==temp)].iloc[0]
            record(f'II_Heating1_{mode}_FWHM_{temp}K',row['width_fwhm_cm-1'],val,2,'cm^-1','Main p9')
    for mode,temp,other,u_report in [('G',280,'Spikes_Removed_UP_2',.06),('D',270,'Spikes_Removed_DOWN_1',.23)]:
        first=ii[(ii.sequence=='Spikes_Removed_UP_1')&(ii.peak_id==mode)&(ii.temperature_K==temp)].iloc[0]
        second=ii[(ii.sequence==other)&(ii.peak_id==mode)&(ii.temperature_K==temp)].iloc[0]
        record(f'II_{mode}_{temp}K_cycle_offset',second['position_cm-1']-first['position_cm-1'],.85,2,'cm^-1','Main p8; SI p15')
        record(f'II_{mode}_{temp}K_cycle_offset_uncertainty',math.hypot(first['position_std_cm-1'],second['position_std_cm-1']),u_report,2,'cm^-1','Main p8; SI p15')
    pd.DataFrame(checks).to_csv(out/'reported_results_verification.csv',index=False)
    failures=[r for r in checks if r['matches_reported_precision'] is False]
    (out/'verification.json').write_text(json.dumps({'checks':len(checks),'display_mismatches':failures,'accepted_table_rows':len(joined)},indent=2),encoding='utf-8')
    print(f'Checked {len(checks)} quantities; {len(failures)} differences from printed precision (retained in report).')
    if failures:
        raise SystemExit('Reported quantities differ; inspect verification.json.')

if __name__=='__main__':main()
