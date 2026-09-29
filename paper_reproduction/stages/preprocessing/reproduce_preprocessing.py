# SPDX-License-Identifier: CC-BY-4.0
"""Reproduce archived spectrum processing without modifying the inputs."""
from pathlib import Path
import os
import argparse,json,os
import numpy as np
import pandas as pd
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

ROOT=Path(os.environ["RAMAN_DATA_ROOT"]).resolve()

def prepare_output_directory(path):
    output = Path(path).resolve()
    bases = [ROOT / "outputs", ROOT / "_verification"]
    if any(base.resolve() != base for base in bases):
        raise ValueError("Output roots must not be redirected by filesystem links.")
    if not any(output.is_relative_to(base) for base in bases):
        raise ValueError("Output must be inside release outputs/ or _verification/.")
    output.mkdir(parents=True, exist_ok=True)
    for directory, subdirectories, files in os.walk(output, followlinks=False):
        for name in subdirectories + files:
            item = Path(directory) / name
            if item.resolve() != item:
                raise ValueError("Output directory contains a redirected filesystem path.")
    return output


def filesystem_path(path):
    if os.name=='nt' and not str(path).startswith('\\\\?\\'):
        return Path('\\\\?\\'+str(path.resolve()))
    return path

def read_matrix(path):
    path=filesystem_path(path)
    with path.open(encoding='utf-8-sig') as stream:
        line=stream.readline()
    try:
        [float(v) for v in line.split()]
        header=None
    except ValueError:
        header=0
    frame=pd.read_csv(path,sep='\t' if '\t' in line else r'\s+',header=header).dropna(axis=1,how='all')
    return frame.to_numpy(float)

def baseline_correct(y,lam,p,n):
    # Same sparse ALS equations and comparison convention as the retained source.
    if not np.isfinite(y).all():
        raise ValueError('Non-finite source value: no replacement is made by this release runner.')
    size=len(y)
    d=diags([np.ones(size),-2*np.ones(size),np.ones(size)],[0,-1,-2],shape=(size,size-2),format='csc')
    penalty=(lam*d.dot(d.T)).tocsc()
    weights=np.ones(size)
    for _ in range(n):
        z=spsolve(diags(weights,0,shape=(size,size),format='csc')+penalty,weights*y)
        weights=p*(y>z)+(1-p)*(y<=z)
    return y-z

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/preprocessing')
    args=parser.parse_args();out=prepare_output_directory(args.output)
    records=json.loads((ROOT/'metadata/preprocessing_provenance.json').read_text(encoding='utf-8'))
    means=json.loads((ROOT/'metadata/temperature_mean_provenance.json').read_text(encoding='utf-8'))
    edits=pd.read_csv(ROOT/'metadata/spike_correction_deltas.csv')
    selection=pd.read_csv(ROOT/'metadata/spectrum_selection.csv')
    checks=[];cache={}
    for rec in records:
        raw=read_matrix(ROOT/rec['raw_release']);actual_clean=read_matrix(ROOT/rec['spike_cleaned_release'])
        for row in edits.loc[edits.spike_cleaned_file==rec['spike_cleaned_release']].itertuples():
            i=int(row.row_index_0based);j=int(row.column_number_1based)-1
            if raw[i,j]!=row.raw_value:raise ValueError('Raw value differs from observed historical change record.')
            raw[i,j]=row.stored_processed_value
        if not np.array_equal(raw,actual_clean):raise ValueError('Historical spike correction replay differs.')
        chosen=raw[:,np.array(rec['selected_columns_1based'])-1]
        if not np.array_equal(chosen,read_matrix(ROOT/rec['selected_spike_cleaned_release'])):
            raise ValueError('Historical column selection differs.')
        cropped=chosen[(chosen[:,0]>=rec['crop_min'])&(chosen[:,0]<=rec['crop_max'])]
        corrected=np.column_stack([cropped[:,0]]+[baseline_correct(cropped[:,i],rec['als_lambda'],rec['als_p'],rec['als_iterations']) for i in range(1,cropped.shape[1])])
        expected=read_matrix(ROOT/rec['baseline_corrected_release'])
        err=float(np.max(np.abs(corrected-expected)))
        if not np.allclose(corrected,expected,rtol=0,atol=1e-7):raise ValueError('ALS output differs beyond sparse-solver tolerance.')
        cache[rec['baseline_corrected_release']]=corrected
        dest=out/'baseline_corrected'/rec['family']/rec['sequence']/Path(rec['baseline_corrected_release']).name
        dest.parent.mkdir(parents=True,exist_ok=True)
        np.savetxt(filesystem_path(dest),corrected,delimiter='\t',fmt='%.17g')
        checks.append({'stage':'spike_selection_ALS','source':rec['raw_release'],'max_abs_difference':err,'verified':True})
    for rec in means:
        sub=selection[(selection.family==rec['family'])&(selection.sequence==rec['sequence'])&(selection.temperature_K==rec['temperature_K'])&selection.included_for_all_trend_peaks.astype(bool)]
        ys=[];x=None
        for row in sub.itertuples():
            data=cache[row.source_release_path]
            if x is None:x=data[:,0]
            if not np.array_equal(x,data[:,0]):raise ValueError('Common measured Raman grid is required; no resampling is performed.')
            ys.append(data[:,int(row.y_column_number)-1])
        values=np.column_stack([x,np.mean(ys,axis=0)])
        dest=out/'temperature_mean'/rec['family']/rec['sequence']/Path(rec['release_path']).name
        dest.parent.mkdir(parents=True,exist_ok=True)
        pd.DataFrame(values,columns=['Column 1','Column 2']).to_csv(filesystem_path(dest),sep='\t',index=False,float_format='%.10g')
        actual=read_matrix(dest);expected=read_matrix(ROOT/rec['release_path'])
        err=float(np.max(np.abs(actual-expected)))
        if not np.allclose(actual,expected,rtol=0,atol=1e-7):raise ValueError('Temperature mean differs beyond numerical export tolerance.')
        checks.append({'stage':'temperature_mean','source':rec['release_path'],'n_columns':len(ys),'max_abs_difference':err,'exact_array':bool(np.array_equal(actual,expected)),'verified':True})
    (out/'verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(f'Verified {len(records)} processing chains and {len(means)} temperature means.')

if __name__=='__main__':main()
