# Noise Pollution Estimation Methodology

## 1. CoRTN Traffic Acoustic Model

Noise pollution estimation utilizes the Calculation of Road Traffic Noise (CoRTN) acoustic formulation to estimate sound level proxies $L_{eq}$ (dBA) based on vehicle composition and density:

$$L_{eq, \text{proxy}} = 10 \log_{10}\left( \sum_{i} N_i \cdot 10^{0.1 \cdot L_{ref, i}} \right) - 20 \log_{10}\left( \frac{d}{d_{ref}} \right)$$

### Acoustic Vehicle Weights

| Vehicle Category | Acoustic Weight | Relative Sound Proxy ($L_{ref}$) |
| :--- | :---: | :---: |
| **Motorcycle** | 10.0 | 78.0 dBA |
| **Heavy Truck** | 8.0 | 76.0 dBA |
| **Bus** | 6.0 | 74.0 dBA |
| **Light Commercial Van** | 1.8 | 62.0 dBA |
| **Passenger Car** | 1.0 | 58.0 dBA |
| **Electric Vehicle (EV)** | 0.2 | 42.0 dBA |

---

## 2. Relative Noise Pollution Index (0 to 100 Scale)

The Relative Noise Pollution Index maps $L_{eq, \text{proxy}}$ onto a 0-100 scale:

$$\text{Noise Index} = \min\left(100, \max\left(0, \frac{L_{eq, \text{proxy}} - 40}{50} \times 100\right)\right)$$

---

## ⚠️ Scientific Limitation Disclaimer

The noise values presented by this camera system are **relative acoustic indices** derived from CoRTN traffic composition models. They do **NOT** represent physical calibrated microphone sound pressure levels (dB SPL). Calibrated sound pressure measurement requires external precision decibel meters.
