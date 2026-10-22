# Keras Tuner toy results

Algorithm used: `hyperband`

## Best hyperparameters

- units: 224
- learning_rate: 0.0003

## All trials

| trial_id | units | learning_rate | epochs_trained | best_val_accuracy |
| --- | ---: | ---: | ---: | ---: |
| 0005 | 224 | 0.0003 | 6 | 0.849333 |
| 0009 | 128 | 0.001 | 6 | 0.842333 |
| 0008 | 192 | 0.01 | 6 | 0.827667 |
| 0006 | 160 | 0.01 | 6 | 0.823667 |
| 0000 | 224 | 0.0003 | 2 | 0.822333 |
| 0004 | 160 | 0.01 | 2 | 0.820667 |
| 0003 | 224 | 0.003 | 2 | 0.820333 |
| 0002 | 160 | 0.001 | 2 | 0.818000 |
| 0001 | 96 | 0.0003 | 2 | 0.811667 |
| 0007 | 32 | 0.0001 | 6 | 0.784333 |

Retrained best model validation accuracy after 3 epochs: 0.833667
