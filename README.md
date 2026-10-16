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
- GAT idea: basically a camera build where each plant is a big pixel looking at the sky, position? no matter, the graph will learn the weights on the edges reconstructing the 
- the battery idea: batteries change the behaviour of PV dramatically
    - less benefit during the day (concurrent houshold usage + battery charging)
    - more benefit during the night, summed with the average lower night consumptions, this makes the power request to the grid almost drop to zero => this makes for a naïve heuristic engineered features that can be  used as an input to increase the predictive ability
    - ...also clustering could be used to cluster PV TRUE in 2 groups
