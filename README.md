# NeuralWave - Dropped Neurons

## Structure

```
nw26_dropped_neurons
├── assets/    # images, logos, media
├── docs/      # notes, pitch, architecture
├── src/       # main application code
└── tests/     # test suite
```

## Active work Timeline:
 - look at the dataset
 - understand the data physics
 - first PRESENT/NOT PRESENT baseline:
    - 93% Logistic Regression [TASK1][67 points achieved] 
    - we can use this to cluster the unlabeled:
        -   244 found pv
        - 1004 found no pv
        - we can train predictive model on both: the power prediction difference would be a produced power prediction estimator [extra 15 points]
- improvements: LSTM -> GAT
