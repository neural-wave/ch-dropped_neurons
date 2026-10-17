# NeuralWave - Dropped Neurons

## Structure

```
nw26_dropped_neurons
├── assets/    # images, logos, media
├── docs/      # notes, pitch, architecture
├── src/       # main application code
└── tests/     # test suite
```



## What features are hidden in the data:
 - before and after (we know when solar panels are added to a house)
 - relational information: can other past serieses help forese the node future?
 - labeling the unlabeled data

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

- cross checking unsupervised clustering and LSTM classifier:
    - run 2-NN (should we try SVN?) clustering on both labeld and unlabled data together
    - check if the clusters actually separate the labeled data (f1 .85)
    - now we can apply LSTM to unlabeled data and see if matches the clustering prediction (how much worse than .85?) 
    - if it does well we are golder, if it does bad, it's not necessarily fault of the LSTM (could be the clustering)

- other option: cross check LSTM against US households
- CHECKUP:
    - what can we do now?
        - the visual representation of the spectrum gives pv date installation away
            - this has huge consequences:
                - manually label unlabeled data to have a better dataset
                - then train on more labeled data
                - this will help the harder model not trained on change point, but only split before/after behaviour
                - simple logical mixture of experts idea:

    - what does not work ?
        - f1 scores on labaled data in unreliable (few data)
            - >proposed solution: cross check on PV data from us households
        - we are detecting a change point in power usage: some users start with PV installed
            - >this could be fixed by the extra manually labeled data


```
NEXT STEPS:
-> try the CNN segmenter to get the start dates on our unlabeled data
-> manually review
-> if it works we are golden
-> otherwise we could pretreain on the large labeled dataset we found and retry
```