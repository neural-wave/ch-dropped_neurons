# PV Detection from Consumption Profiles

## The Challenge
- AIL needs to know which customers have rooftop solar, when it was installed, and how large it is.
- The only input is 15-minute consumption data. No weather, location, or irradiation data may be used.

## Why It Matters
- Accurate PV records keep customer information current.
- Energy monitoring improves without installing new metering.
- Detection relies on data AIL already collects.

## The Fingerprint in the Data
- Solar generation creates a midday dip in grid withdrawals.
- The dip follows the seasons: deeper in summer, shallower in winter.
- From the commissioning date onward, the daily load shape changes persistently.
- Extensions appear as step changes in the size of the dip.

## The Assets
- 40 labelled cases calibrate and validate the approach.
- 1,148 unlabelled series show typical load behaviour across customers.
- Calendar features and epoch timestamps keep the time structure correct, including daylight saving days.

## The Idea: A Graph Attention Network
- A GAT links customers with similar load patterns.
- Each customer's next value is predicted from its own history and its neighbours' values.
- Edges come from consumption similarity only, not geography.

## Why It Works
- Neighbours capture the shared environment, including weather and seasonal effects.
- The residual (actual minus predicted) isolates what neighbours can't explain.
- A sustained negative residual from a specific date is a candidate commissioning date.

## How It Plugs In
- Residual statistics and neighbour-relative features feed the presence classifier.
- The same features inform the date and capacity estimators.
- Pretraining uses the unlabelled series, so labelled cases are reserved for tuning.

## Risks and Safeguards
- **Over-smoothing:** If the GAT predicts a customer too well, it also learns the PV effect and erases the signal. Check residuals on labelled cases.
- **Leakage:** Hidden test labels must never enter training or threshold tuning.
- **Deployment:** The pipeline must run offline from `inference.py`, with weights saved as artifacts.

## The Payoff
- A reproducible pipeline that prioritises detection accuracy, which carries the largest share of the score.
- A foundation for later change-point work, including battery detection.