"""Check refiner parameter export, customer independence, and output validity."""
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from lstm_pipeline import load_series,read_labels,write_outputs
from nw26_dropped_neurons.src.simple_lstm_model.pv_refinement import fit_refiner,refine_predictions,save_refiner,load_refiner
from nw26_dropped_neurons.src.simple_lstm_model.verify_lstm import fixture

def run():
 with tempfile.TemporaryDirectory() as folder:
  root=Path(folder)
  frame=pd.concat([fixture('0001','2024-03-01','2024-05-01'),fixture('0002','2024-03-01','2024-05-01')])
  frame.to_csv(root/'input.csv',sep=';',index=False)
  series,_=load_series(root/'input.csv',chunksize=1000)
  labels=pd.DataFrame({'customer_id':['0001','0002'],'pv_installed':[1,0],
   'pv_installation_date':[pd.Timestamp('2024-04-01'),pd.NaT],'pv_capacity_kwp':[10.,np.nan]}).set_index('customer_id')
  refiner=fit_refiner(series,labels)
  base=pd.DataFrame({'customer_id':['0001','0002'],'pv_installed':[True,False],
   'pv_installation_date':['2024-04-01',''],'pv_capacity_kwp':[10.,np.nan],'pv_probability':[.9,.1]})
  a=refine_predictions(series,base,refiner)
  save_refiner(refiner,root/'refiner.pt')
  loaded=load_refiner(root/'refiner.pt')
  b=refine_predictions(series,base,loaded)
  pd.testing.assert_frame_equal(a,b)
  a_only=refine_predictions(series[:1],base.iloc[:1].copy(),loaded)
  pd.testing.assert_frame_equal(a.iloc[:1],a_only)
  assert len(a)==2 and set(a.customer_id)=={'0001','0002'}
  assert a.iloc[0].pv_capacity_kwp>0
  write_outputs(a,root)
  false=a[a.customer_id=='0002'].iloc[0]
  assert false.pv_installation_date=='' and np.isnan(false.pv_capacity_kwp)
 print('PASS: tree export, safe refiner reload, one-positive training, customer-independent inference, leading-zero IDs and valid outputs.')

if __name__=='__main__':run()
