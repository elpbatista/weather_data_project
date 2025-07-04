# Integration with: *Short-term Weather Forecasting using Spatial Feature Attention based LSTM Model*

This project is inspired by and adapts the methodology from the paper:

> Suleman, M. A. R., & Shridevi, S. (2022). Short-term weather forecasting using spatial feature attention based LSTM model. *IEEE Access*. <https://doi.org/10.1109/ACCESS.2022.3196381>

---

## Objective

To replicate and extend the SFA-LSTM model architecture for temperature forecasting using hourly weather data from multiple cities in Oregon's Willamette Valley.

---

## Comparison with the Original Paper

| Aspect                      | Original Paper                                                                  | This Project                                                                 |
|----------------------------|----------------------------------------------------------------------------------|------------------------------------------------------------------------------|
| Model Type                 | Spatial Feature Attention-based LSTM (SFA-LSTM)                                  | Implemented SFA-LSTM using TensorFlow/Keras                                 |
| Architecture               | Encoder-decoder with attention in decoder                                       | Attention applied before LSTM; streamlined design                           |
| Input Features             | Dew Point, Wind Chill, RH, Station & Sea Pressure, Wind Speed                   | Same variables, sourced from Open-Meteo                                     |
| Target Variable            | Temperature                                                                     | Temperature                                                                 |
| Sequence Length            | 24 hours                                                                        | Configurable; 24-hour window default                                        |
| Evaluation Metrics         | MSE, MAE, R²                                                                     | Same (logged and visualized)                                                |
| Data Source                | WeatherStats (Saskatoon, Canada)                                                | Open-Meteo (Salem, Eugene, Corvallis – Oregon, USA)                         |
| Station vs. Combined       | Single station                                                                  | Trained both individual city models and a combined multi-station model      |
| Visualization              | Attention map (Fig. 11), Feature correlation (Table VI)                         | Performance plots, residuals, error distributions                           |

---

## Contributions

- Reproducible ETL & training pipeline: Data cleaning, sequence preparation, model training, and evaluation
- Multi-location modeling: Supported both individual and combined station training
- Model persistence: Models saved in `.keras` format for reuse
- Advanced visualizations: Included residuals and distribution plots across all models

---

## Planned Additions

- [ ] Attention weight visualization (like Fig. 11)
- [ ] Feature correlation analysis (to replicate Table VI)
- [ ] Multi-horizon forecasting

---

## Citation

If referencing this project, please cite the original source paper:

> Suleman, M. A. R., & Shridevi, S. (2022). *Short-term weather forecasting using spatial feature attention based LSTM model*. IEEE Access. <https://doi.org/10.1109/ACCESS.2022.3196381>
