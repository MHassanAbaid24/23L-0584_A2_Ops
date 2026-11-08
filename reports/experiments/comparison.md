# Experiment comparison (sorted by val ROC-AUC; selection uses validation only)

| run             |   best_epoch |   threshold |   val_auc |   val_f1 |   val_precision |   val_real→fake |   val_fake→real |   test_acc |   test_precision |   test_recall |   test_f1 |   test_auc |   test_real→fake |
|:----------------|-------------:|------------:|----------:|---------:|----------------:|----------------:|----------------:|-----------:|-----------------:|--------------:|----------:|-----------:|-----------------:|
| E5-mobilenetv2  |           33 |      0.6725 |    0.8807 |   0.7832 |          0.9032 |          0.0923 |          0.3086 |     0.7823 |           0.9016 |        0.6790 |    0.7746 |     0.8709 |           0.0909 |
| E7-ft-lr-1e-4   |           24 |      0.5000 |    0.8786 |   0.8250 |          0.8354 |          0.2000 |          0.1852 |     0.8163 |           0.8214 |        0.8519 |    0.8364 |     0.8902 |           0.2273 |
| E2-no-augment   |           34 |      0.3819 |    0.8764 |   0.8471 |          0.8090 |          0.2615 |          0.1111 |     0.8027 |           0.7708 |        0.9136 |    0.8362 |     0.8915 |           0.3333 |
| E0-base         |           35 |      0.6108 |    0.8697 |   0.7895 |          0.8451 |          0.1692 |          0.2593 |     0.8027 |           0.8421 |        0.7901 |    0.8153 |     0.8853 |           0.1818 |
| E1-base-autolog |           34 |      0.6016 |    0.8663 |   0.7922 |          0.8356 |          0.1846 |          0.2469 |     0.8095 |           0.8354 |        0.8148 |    0.8250 |     0.8844 |           0.1970 |
| E6-resize-pad   |           34 |      0.5000 |    0.8587 |   0.7975 |          0.8182 |          0.2154 |          0.2222 |     0.8299 |           0.8256 |        0.8765 |    0.8503 |     0.8883 |           0.2273 |
| E3-frozen-only  |           20 |      0.4907 |    0.8572 |   0.8324 |          0.7826 |          0.3077 |          0.1111 |     0.7687 |           0.7527 |        0.8642 |    0.8046 |     0.8739 |           0.3485 |
| E4-img160       |           31 |      0.5550 |    0.8298 |   0.7826 |          0.7875 |          0.2615 |          0.2222 |     0.8027 |           0.8250 |        0.8148 |    0.8199 |     0.8694 |           0.2121 |
| cnn-baseline    |           11 |      0.5422 |    0.6957 |   0.6711 |          0.7183 |          0.3077 |          0.3704 |     0.6871 |           0.7536 |        0.6420 |    0.6933 |     0.7421 |           0.2576 |

`val_precision` is the precision of the 'AI' verdict on the validation split at each run's chosen threshold, taken from the run's `metrics.json`. An earlier version of this table copied MLflow's `val_precision`, which was the last training epoch's Keras metric at threshold 0.5 (a name collision, fixed in `train.py` by logging the thresholded values as `val_thr_*`).
