# Scientific Methodology & Emission Factor Formulation

## 1. Scientific Limitation & Nomenclature

> **CRITICAL SCIENTIFIC PRINCIPLE:**
> Computer vision systems operating standard RGB cameras **cannot** physically measure gas molecular concentrations ($PM_{2.5}, PM_{10}, NO_x, CO, CO_2, O_3$).
> Therefore, all camera outputs in this platform are designated strictly as **"Estimated Vehicle Pollution Contribution"** ($g/hr$) or **"Relative Noise Pollution Index ($dB_A$)"**.

Physical ambient pollutant concentrations are fetched via dedicated environmental sensor station adapters (OpenAQ and Central Pollution Control Board CPCB India).

---

## 2. Air Pollution Calculation Model (COPERT V / EEA Standard)

Traffic air pollution contribution is calculated using standard emission factor models:

$$E_i = \sum_{j} \left( N_j \cdot EF_{i,j}(v) \cdot d_j \right)$$

Where:
- $E_i$: Total mass emission rate of pollutant $i$ ($g/hr$).
- $N_j$: Number of tracked vehicles of vehicle class $j$ (Car, Bus, Truck, Motorcycle).
- $EF_{i,j}(v)$: Speed-dependent emission factor ($g/km$) for pollutant $i$, vehicle class $j$, and fuel type $k$ (Petrol, Diesel, EV, CNG, Hybrid).
- $d_j$: Distance traveled ($km$) per hour based on velocity $v$ ($km/h$).

### Emission Factors Table ($g/km$)

| Vehicle Class | Fuel Type | $PM_{2.5}$ | $PM_{10}$ | $NO_x$ | $CO$ | $CO_2$ |
|---|---|---|---|---|---|---|
| Passenger Car | Petrol | 0.003 | 0.005 | 0.250 | 1.500 | 140.0 |
| Passenger Car | Diesel | 0.035 | 0.045 | 0.650 | 0.400 | 130.0 |
| Passenger Car | EV | 0.001 | 0.002 | 0.000 | 0.000 | 0.000 |
| Transit Bus | Diesel | 0.150 | 0.190 | 4.500 | 2.100 | 750.0 |
| Heavy Truck | Diesel | 0.180 | 0.220 | 5.200 | 2.500 | 850.0 |

---

## 3. Relative Noise Propagation Model (CoRTN)

Road traffic noise is modeled according to the Calculation of Road Traffic Noise (CoRTN) acoustic standard:

$$L_{eq} = 10 \log_{10} \left( \sum_{m} 10^{\frac{L_{m} - \Delta L_{dist}}{10}} \right)$$

Where:
- $L_m$: Base acoustic sound power level ($dB_A$) at reference 10 meters distance.
- $\Delta L_{dist} = 20 \log_{10}(R / 10)$: Distance attenuation factor at distance $R$ meters.
- $L_{eq}$: Equivalent continuous sound pressure level ($dB_A$).
