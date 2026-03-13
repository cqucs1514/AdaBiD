# AdaBiD

PyTorch reproduction of the paper:

> **AdaBiD: An Efficient Traffic Forecasting Model Based on Adaptive Graph Construction and Bidirectional Diffusion Convolution**  
> Wengang Li, Yumang Liu, Xiao Wu, Yi Tang, Dihua Sun, Linjiang Zheng  
> Preprint submitted to Elsevier

---

## Project Structure

```
AdaBiD/
├── data/
│   └── PEMS08.npz          ← place your PEMS .npz file here
├── checkpoints/             ← best model weights saved here (auto-created)
├── logs/                    ← training logs (auto-created)
├── dataset.py               ← data loading & Z-score normalisation
├── model.py                 ← AdaBiD architecture
├── train.py                 ← training & evaluation script
└── utils.py                 ← MAE / RMSE / MAPE metrics, StandardScaler
```



