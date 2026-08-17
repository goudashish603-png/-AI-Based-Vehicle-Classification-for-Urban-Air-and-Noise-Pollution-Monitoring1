# Future Work & Enhancements

## Recommended System Extensions

1. **License Plate Recognition (ALPR)**:
   - Integrate Automatic License Plate Recognition (ALPR) to query national vehicle registries (e.g. VAHAN in India or DVLA in the UK) for exact powertrain specifications and Euro emission standard ratings.

2. **AERMOD / CALPUFF Meteorological Dispersion Integration**:
   - Link estimated $g/hr$ vehicle mass emissions with real-time weather data (wind speed, wind direction, boundary layer height, temperature) to model ground-level pollutant dispersion ($µg/m³$).

3. **Multi-Camera Edge Deployment**:
   - Deploy lightweight TensorRT / ONNX pipeline models across edge smart city cameras connected to a centralized GIS map server.

4. **Calibrated Physical Microphone Fusion**:
   - Pair camera feeds with Bluetooth / IoT calibrated sound pressure level (dB SPL) microphones to cross-validate CoRTN acoustic model estimates.
