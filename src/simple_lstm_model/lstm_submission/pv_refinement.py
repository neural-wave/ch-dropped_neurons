"""Consumption-only refinement for the AIL multi-task LSTM.

Model parameters and a shared daytime factor are learned on training customers.
The module never reads development labels at inference.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import gaussian_filter1d
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from lstm_pipeline import predict, load_artifact

NIGHT = [0, 1, 3, 4, 5, 22, 23]


def latent_design(s,harmonics=1,target='noon'):
    kw=np.expm1(s.x[:,:24]).astype(float)*s.scale
    day=kw[:,10:16].mean(axis=1) if target=='day' else kw[:,11:15].mean(axis=1)
    night=kw[:,NIGHT].mean(axis=1)
    phase=s.dates.dayofyear.to_numpy()/365.25*2*np.pi
    H=[np.ones(len(kw)),night/max(np.median(night),.01)]
    for h in range(1,harmonics+1): H += [np.sin(h*phase),np.cos(h*phase)]
    H += [(s.dates.dayofweek.to_numpy()==d).astype(float) for d in range(1,7)]
    return day,np.array(H).T

def learn_template(series,labels,harmonics=1,target='noon',iterations=80,smooth=0):
    dates=pd.date_range(min(s.dates[0] for s in series),max(s.dates[-1] for s in series))
    solar=np.zeros(len(dates))
    cases=[]
    for s in series:
        label=labels.loc[s.customer_id]
        if not label.pv_installed: continue
        y,H=latent_design(s,harmonics,target)
        offsets=(s.dates-dates[0]).days.to_numpy()
        active=(s.dates>=label.pv_installation_date).astype(float)
        cap=label.pv_capacity_kwp
        weight=(s.quality>.8).astype(float)
        inv=np.linalg.pinv(H.T@(weight[:,None]*H)+1e-4*np.eye(H.shape[1]))
        cases.append((y,H,offsets,active,cap,weight,inv))
    # Train-only coordinate descent for Y_i = baseline_i - capacity_i * active_i * S.
    # Calendar and contemporaneous nighttime load are the only baseline predictors.
    for _ in range(iterations):
        num=np.zeros(len(dates));den=np.zeros(len(dates))
        for y,H,ix,active,cap,weight,inv in cases:
            baseline=H@(inv@(H.T@(weight*(y+cap*active*solar[ix]))))
            residual=(baseline-y)/cap
            num[ix] += active*weight*residual
            den[ix] += active*weight
        updated=np.divide(num,den,out=np.zeros_like(num),where=den>0)
        updated=np.maximum(updated,0)
        if smooth: updated=gaussian_filter1d(updated,smooth)
        solar=.5*solar+.5*updated
    return dates,solar

def shared_candidate(s,template,harmonics=1,target='noon',min_side=7):
    dates,solar=template
    offsets=(s.dates-dates[0]).days.to_numpy()
    g=np.zeros(len(s.dates))
    covered=(offsets>=0)&(offsets<len(dates))
    g[covered]=solar[offsets[covered]]
    y,H=latent_design(s,harmonics,target)
    weight=(s.quality>.8).astype(float)
    inv=np.linalg.pinv(H.T@(weight[:,None]*H)+1e-4*np.eye(H.shape[1]))
    residual=y-H@(inv@(H.T@(weight*y)))
    suffix=lambda a:np.cumsum(a[::-1],axis=0)[::-1]
    h=suffix(weight[:,None]*g[:,None]*H)
    num=suffix(weight*g*residual)
    den=suffix(weight*g*g)-np.einsum('ij,jk,ik->i',h,inv,h)
    cap=np.maximum(-num/np.maximum(den,1e-7),.01)
    gain=np.maximum(-num,0)**2/np.maximum(den,1e-7)
    pre=np.cumsum(weight)-weight; post=suffix(weight)
    gain[(pre<min_side)|(post<min_side)]=0
    best=int(gain.argmax())
    noise=max(np.median(np.abs(residual[weight>0]-np.median(residual[weight>0])))*1.4826,.01)
    return {'offset':best,'capacity':float(cap[best]),'strength':float(np.sqrt(gain[best])/noise),
            'gain':gain,'capacities':cap}

def local_candidate(s, seasonal=.55, control='night', harmonics=1, min_side=14):
    kw=np.expm1(s.x[:,:24]).astype(float)*s.scale
    valid=s.quality>.8
    day=kw[:,10:16].mean(axis=1)
    night=kw[:,[0,1,3,4,5,22,23]].mean(axis=1)
    doy=s.dates.dayofyear.to_numpy()/365.25*2*np.pi
    # Calendar harmonic; phase is a hyperparameter, not external irradiation.
    g=1+seasonal*np.cos(doy-2*np.pi*172/365.25)
    controls=[np.ones(len(kw))]
    if control in ['night','difference','ratio']:
        controls += [night/max(np.median(night),.01)]
    if control=='difference': day=day-night
    if control=='ratio': day=day/np.maximum(night,s.scale*.1)
    for h in range(1,harmonics+1):
        controls += [np.sin(doy*h),np.cos(doy*h)]
    controls += [(s.dates.dayofweek.to_numpy()==d).astype(float) for d in range(1,7)]
    H=np.array(controls).T
    weight=valid.astype(float)
    scale=max(np.median(day),.01)
    day=day/scale
    # Robustly suppress isolated outages/holiday spikes without erasing steps.
    for _ in range(3):
        inv=np.linalg.pinv(H.T@(weight[:,None]*H)+1e-5*np.eye(H.shape[1]))
        beta=inv@(H.T@(weight*day))
        residual=day-H@beta
        mad=max(np.median(np.abs(residual[valid]-np.median(residual[valid])))*1.4826,.01)
        weight=valid*np.minimum(1,2.5*mad/np.maximum(np.abs(residual),1e-8))
    inv=np.linalg.pinv(H.T@(weight[:,None]*H)+1e-5*np.eye(H.shape[1]))
    residual=day-H@(inv@(H.T@(weight*day)))
    suffix=lambda a: np.cumsum(a[::-1],axis=0)[::-1]
    h=suffix(weight[:,None]*g[:,None]*H)
    numerator=suffix(weight*g*residual)
    denominator=suffix(weight*g*g)-np.einsum('ij,jk,ik->i',h,inv,h)
    amplitude=-numerator/np.maximum(denominator,1e-6)
    gain=np.maximum(-numerator,0)**2/np.maximum(denominator,1e-6)
    pre=np.cumsum(valid)-valid
    post=suffix(valid)
    gain[(pre<min_side)|(post<min_side)]=0
    if gain.max()<=0: best=0
    else: best=int(np.argmax(gain))
    noise=max(np.median(np.abs(residual[valid]-np.median(residual[valid])))*1.4826,.01)
    return {'offset':best,'amplitude':float(max(amplitude[best]*scale,0)),
            'strength':float(np.sqrt(gain[best])/noise),'gain':gain,'amplitudes':amplitude*scale,
            'noise':noise*scale}

def capacity_features(s,offset):
    raw=np.expm1(s.x[:,:24]).astype(float)*s.scale
    n=len(raw);d=int(np.clip(offset,0,n-1))
    before=raw[max(0,d-365):max(d,1)]
    after=raw[d:min(n,d+180)]
    nearbefore=raw[max(0,d-60):max(d,1)]
    nearafter=raw[d:min(n,d+60)]
    phase=s.dates[d].dayofyear/365.25*2*np.pi
    values=[np.log1p(s.scale),np.sin(phase),np.cos(phase),min(d,365)/365,min(n-d,365)/365]
    for data in [before,after,nearbefore,nearafter]:
        for hours in [slice(10,16),slice(11,15),[0,1,3,4,5,22,23],slice(6,10),slice(16,20)]:
            x=data[:,hours].mean(axis=1)
            values.extend(np.log1p(np.quantile(x,[.05,.25,.5,.75,.95])))
        values.extend((data[:,9:17]<s.scale*.05).mean(axis=0))
    for a,b in [(before,after),(nearbefore,nearafter)]:
        diff=np.median(a,axis=0)-np.median(b,axis=0)
        values.extend(np.arcsinh(diff/s.scale))
    y,H=latent_design(s,harmonics=2,target='noon')
    valid=(s.quality>.8)&(np.arange(n)<d)
    if valid.sum()>=14:
        ridge=np.eye(H.shape[1])*.01
        beta=np.linalg.solve(H[valid].T@H[valid]+ridge,H[valid].T@y[valid])
        residual=(H@beta-y)[d:min(n,d+180)]
        values.extend(np.arcsinh(np.quantile(residual,[.1,.25,.5,.75,.9,.95,.99])/s.scale))
        values.extend([np.log1p(max(np.quantile(residual,q),0)) for q in [.5,.75,.9,.95,.99]])
    else: values.extend([0.]*12)
    values.extend(np.log1p(np.quantile(raw[:,11:15].mean(1),[.05,.25,.5,.75,.95])))
    return np.nan_to_num(values).astype(float)



def fit_refiner(series, labels):
    """Fit only on the supplied training-customer labels and consumption."""
    template = learn_template(series, labels, harmonics=2, target='noon')
    grid, solar = template
    available = np.zeros(len(grid), dtype=bool)
    positives = []
    for s in series:
        label = labels.loc[s.customer_id]
        if not label.pv_installed:
            continue
        positives.append(s)
        ix = (s.dates - grid[0]).days.to_numpy()
        available[ix[s.dates >= label.pv_installation_date]] = True
    if not positives or not available.any():
        raise ValueError('The refiner requires PV-positive training customers.')
    # Calendar extrapolation covers days with no active training installation.
    phase = grid.dayofyear.to_numpy() / 365.25 * 2 * np.pi
    H = np.column_stack([np.ones(len(grid)), np.sin(phase), np.cos(phase),
                         np.sin(2 * phase), np.cos(2 * phase)])
    calendar_coef = np.linalg.lstsq(H[available], solar[available], rcond=None)[0]
    solar = np.where(available, solar, np.maximum(H @ calendar_coef, 0))
    X = np.stack([capacity_features(s, (labels.loc[s.customer_id, 'pv_installation_date'] - s.dates[0]).days)
                  for s in positives])
    target = np.log([labels.loc[s.customer_id, 'pv_capacity_kwp'] for s in positives])
    capacities = np.exp(target)
    expected_p = np.maximum(0, 1-2*np.abs(capacities[:, None]-capacities[None, :])
                            / capacities[:, None]).mean(axis=0)
    capacity_prior = float(capacities[expected_p.argmax()])
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1000)).fit(X, target)
    forest = ExtraTreesRegressor(n_estimators=200, min_samples_leaf=2, max_features=.7,
                                 random_state=42, n_jobs=2).fit(X, target)
    trees = []
    for estimator in forest.estimators_:
        t = estimator.tree_
        trees.append({'left': t.children_left.copy(), 'right': t.children_right.copy(),
                      'feature': t.feature.copy(), 'threshold': t.threshold.copy(),
                      'value': t.value[:, 0, 0].copy()})
    result = {'template_start': grid[0].strftime('%Y-%m-%d'), 'solar': solar,
              'calendar_coef': calendar_coef, 'ridge_mean': ridge[0].mean_,
              'ridge_scale': ridge[0].scale_, 'ridge_coef': ridge[1].coef_,
              'ridge_intercept': float(ridge[1].intercept_), 'trees': trees,
              'capacity_prior': capacity_prior, 'capacity_shrinkage': 0.5}
    # Verify our small dependency-independent tree evaluator before export.
    reference = forest.predict(X)
    restored = np.array([_forest_predict(row, trees) for row in X])
    np.testing.assert_allclose(reference, restored, rtol=1e-7, atol=1e-7)
    return result


def _forest_predict(x, trees):
    # sklearn tree input is float32; match its threshold comparisons exactly.
    x = np.asarray(x, dtype=np.float32)
    values = []
    for t in trees:
        node = 0
        while t['left'][node] >= 0:
            node = int(t['left'][node] if x[int(t['feature'][node])] <= t['threshold'][node]
                       else t['right'][node])
        values.append(t['value'][node])
    return float(np.mean(values))


def refine_predictions(series, base, refiner):
    result = base.copy().set_index('customer_id')
    grid = pd.date_range(refiner['template_start'], periods=len(refiner['solar']))
    template = (grid, refiner['solar'])
    for s in series:
        cid = s.customer_id
        if not bool(result.loc[cid, 'pv_installed']):
            result.loc[cid, 'pv_installation_date'] = ''
            result.loc[cid, 'pv_capacity_kwp'] = np.nan
            continue
        shared = shared_candidate(s, template, harmonics=2, target='noon')
        local = local_candidate(s, seasonal=.55, control='ratio', harmonics=0, min_side=7)
        base_date = pd.Timestamp(result.loc[cid, 'pv_installation_date'])
        offsets = [(base_date-s.dates[0]).days, shared['offset'], local['offset']]
        date = int(np.clip(np.median(offsets), 0, len(s.dates)-1))
        # Capacity uses the shared-factor event; date consensus has its own decoder.
        x = capacity_features(s, shared['offset'])
        normalized = (x-refiner['ridge_mean']) / refiner['ridge_scale']
        ridge_log = float(normalized @ refiner['ridge_coef'] + refiner['ridge_intercept'])
        forest_log = _forest_predict(x, refiner['trees'])
        capacities = [float(result.loc[cid, 'pv_capacity_kwp']), shared['capacity'],
                      np.exp(np.clip(ridge_log, -3, 12)), np.exp(np.clip(forest_log, -3, 12))]
        capacities = [c for c in capacities if np.isfinite(c) and c > 0]
        result.loc[cid, 'pv_installation_date'] = s.dates[date].strftime('%Y-%m-%d')
        estimate_log = float(np.median(np.log(capacities)))
        shrink = float(refiner.get('capacity_shrinkage', 0.))
        if shrink:
            estimate_log = (1-shrink)*estimate_log + shrink*np.log(refiner['capacity_prior'])
        result.loc[cid, 'pv_capacity_kwp'] = float(np.exp(estimate_log))
    return result.reset_index()


def predict_refined(model, scaler, series, refiner, threshold=.5):
    return refine_predictions(series, predict(model, scaler, series, threshold), refiner)


def save_refiner(refiner, path):
    def tensors(obj):
        if isinstance(obj, np.ndarray): return torch.from_numpy(obj.copy())
        if isinstance(obj, dict): return {k: tensors(v) for k, v in obj.items()}
        if isinstance(obj, list): return [tensors(v) for v in obj]
        return obj
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'format_version': 1, 'parameters': tensors(refiner)}, path)


def load_refiner(path):
    def arrays(obj):
        if torch.is_tensor(obj): return obj.numpy()
        if isinstance(obj, dict): return {k: arrays(v) for k, v in obj.items()}
        if isinstance(obj, list): return [arrays(v) for v in obj]
        return obj
    artifact = torch.load(path, map_location='cpu', weights_only=True)
    if artifact['format_version'] != 1: raise ValueError('Unsupported refiner artifact version.')
    return arrays(artifact['parameters'])


def predict_bundle(artifact_dir, series):
    """Offline inference with separate presence and event checkpoints."""
    artifact_dir = Path(artifact_dir)
    model, scaler, _ = load_artifact(artifact_dir/'pv_lstm.pt')
    presence, presence_scaler, threshold = load_artifact(artifact_dir/'pv_presence.pt')
    base = predict(model, scaler, series, threshold=0.)
    classification = predict(presence, presence_scaler, series, threshold).set_index('customer_id')
    base['pv_installed'] = base.customer_id.map(classification.pv_installed)
    base['pv_probability'] = base.customer_id.map(classification.pv_probability)
    return refine_predictions(series, base, load_refiner(artifact_dir/'pv_refiner.pt'))

