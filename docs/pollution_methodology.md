# Air Pollution Estimation Scientific Methodology

## 1. Emission Factor Database (EEA COPERT V Standards)

Air pollution calculation utilizes scientifically sourced European Environment Agency (EEA) COPERT V emission factors ($g/km$) stored in `data/external/emission_factors.csv`:

$$\text{Estimated Emission Rate } (g/hr) = \text{Vehicle Count} \times \text{Emission Factor } (g/km) \times \text{Speed } (km/h) \times \text{Activity Factor}$$

### Sourced Emission Factors Sample Table ($g/km$ at 30 km/h)

| Vehicle Category | Fuel Type | $PM_{2.5}$ ($g/km$) | $NO_2$ ($g/km$) | $CO$ ($g/km$) | $CO_2$ ($g/km$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Car** | Petrol | 0.002 | 0.040 | 0.450 | 145.0 |
| **Car** | Diesel | 0.025 | 0.420 | 0.150 | 135.0 |
| **Car** | EV | 0.000 | 0.000 | 0.000 | 0.0 |
| **Truck** | Diesel | 0.085 | 1.850 | 0.850 | 650.0 |
| **Bus** | Diesel | 0.095 | 2.100 | 0.950 | 720.0 |

---

## 2. Vehicle Pollution Contribution Index (0 to 100 Scale)

The 0 to 100 Vehicle Pollution Contribution Index normalizes fleet emissions against urban traffic threshold baselines ($E_{base}$):

$$\text{Pollution Index} = \min\left(100, \sum_{p} w_p \cdot \frac{E_p}{E_{base, p}} \times 100\right)$$

Where weights $w_{PM2.5} = 0.45, w_{NO2} = 0.35, w_{CO} = 0.15, w_{SO2} = 0.05$.

---

## ⚠️ Scientific Limitation Disclaimer

This system calculates **model-based estimated vehicle emissions** ($g/hr$ mass rates) derived from traffic counts, vehicle categories, and fuel lookup tables. It does **NOT** measure ambient atmospheric concentrations ($µg/m³$) or Air Quality Index (AQI) directly from camera images. Ambient concentration depends heavily on meteorological dispersion (wind speed, boundary layer height, humidity) and regional background sources.
