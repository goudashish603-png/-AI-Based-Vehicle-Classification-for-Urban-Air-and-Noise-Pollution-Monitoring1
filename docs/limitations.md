# Scientific & Operational Limitations

## ⚠️ Mandatory System Limitations & Constraints

1. **Air Pollution Concentration Limits**:
   - **Constraint:** Optical RGB video cameras cannot directly detect gas molecules ($NO_2, CO, SO_2$) or particulate matter ($PM_{2.5}, PM_{10}$) suspended in the air.
   - **Methodological Solution:** System outputs are explicitly named **Estimated Vehicle Emission Contribution** ($g/hr$ mass emission rate derived from EEA COPERT V traffic factors). Ambient air quality ($µg/m³$) requires physical air sensors or meteorological dispersion modeling (AERMOD/CALPUFF).

2. **Acoustic Noise Measurement Limits**:
   - **Constraint:** Cameras capture light, not sound pressure waves.
   - **Methodological Solution:** System outputs are explicitly named **Relative Noise Pollution Index** derived from CoRTN acoustic traffic composition models. Calibrated decibel (dB SPL) measurements require precision microphones.

3. **Fuel Type Inference Ambiguity**:
   - **Constraint:** Visually identical vehicle body styles (e.g. Ford Focus or VW Golf) are sold in petrol, diesel, and hybrid variants that cannot be distinguished visually from RGB images.
   - **Methodological Solution:** System returns `UNKNOWN` or `AMBIGUOUS` with confidence scoring whenever visual features are ambiguous. Fuel data is never fabricated.

4. **Correlation vs. Causation**:
   - **Constraint:** Statistical correlation between camera traffic counts and nearby ambient air quality monitoring stations does not prove physical causation.
   - **Methodological Solution:** Ambient air concentrations are influenced by wind speed, planetary boundary layer height, temperature inversions, industrial background emissions, and regional transport. All reports display explicit causation warnings.
