# model/

The shipped model lives here: `encoder.pt` (1-D day encoder), `downstream.joblib` (day model, 2-D CNN,
change-point settings) and `fit_info.json`. `inference.py` loads this folder first.

`run_all.sh` writes it (step 5, `--save-to model`). To use a model trained elsewhere, copy those files here, e.g.
from the GPU server (seed-2 run):

```bash
rsync -avz 'ai@<server>:~/Repositories/AIL_data_for_neural_wave_v1.1/pv_ssl/artifacts/model_cnn_seed2/' model/
```

The pretrained conv stack `artifacts/heatcnn_ssl_relsmooth.pt` is only needed for training, not for inference.
