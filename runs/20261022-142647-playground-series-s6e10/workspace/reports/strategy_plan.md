# Strategy — binary_classification on `satisfaction`

The strategy focuses on building a strong gradient boosting baseline, followed by targeted feature engineering based on analyst insights and public notebook patterns. Key areas include service rating interactions, delay-related features, and categorical interactions. The plan rotates through LightGBM, CatBoost, and XGBoost to leverage their different strengths and decorrelated errors for potential later blending. Cross-validation will be stratified 5-fold with a fixed seed.

## Cross-validation
- folds: 5 (stratified: True, seed: 42)
- submission gate: CV improvement >= 0.0001

## Roadmap
| # | title | family | feature ideas |
|---|---|---|---|
| 1 | LightGBM Baseline with Raw Features | LightGBM | - |
| 2 | Feature Engineering: Service Interactions and Delay Features | LightGBM | Service_Rating_Interaction, Delay_Ratio |
| 3 | Feature Engineering: Rating Counts and Interaction Terms | LightGBM | Service_Per_Category_Count |
| 4 | CatBoost with Enhanced Categorical Handling and Interactions | CatBoost | Customer_Type_Travel_Class_Interaction |
| 5 | XGBoost with Target Encoding and Route-based Features | XGBoost | - |
| 6 | Advanced Feature Engineering: Route-based and Interaction Counts | LightGBM | Flight_Distance_Binned, Count_Encoding_Customer_Type, Count_Encoding_Type_of_Travel, Count_Encoding_Class, Route_Mean_Wifi_Service, Route_Mean_Seat_Comfort, Route_Mean_Arrival_Delay, Route_Std_Arrival_Delay, Relative_Wifi_Route, Relative_Seat_Route, Relative_Delay_Route |
| 7 | Target Encoding of Combined Categoricals | XGBoost | TE_FlightDistance_Class, TE_FlightDistance_TypeOfTravel, TE_CustomerType_TypeOfTravel, TE_Class_TypeOfTravel |
| 8 | Comprehensive Service Rating Interactions and Derived Features | CatBoost | Delay_Total, Delay_Diff, Delay_Ratio, Zero_Ratings_Count, Net_Satisfaction_Score, NonZero_Rating_Mean, NonZero_Rating_Std, Wifi_x_Boarding, Comfort_x_Clean, Entertainment_x_Wifi, Wifi_is_zero, Boarding_is_zero, Clean_Food_Ratio, Wifi_Booking_Gap, Digital_Service_Index, Digital_Satisfaction_Gap, Physical_Service_Index, Cabin_Comfort_Cluster, Service_Expectation_Gap, Wifi_Boarding_Ratio, Loyal_Business_Segment |
| 9 | Target Encoding of Combined Categoricals with Flight Distance Bins | XGBoost | Flight_Distance_Binned_10, TE_FlightDistanceBinned_Class, TE_FlightDistanceBinned_TypeOfTravel, TE_CustomerType_TypeOfTravel |
| 10 | Advanced Count and Frequency Encodings | LightGBM | Count_Encoding_Customer_Type, Count_Encoding_Type_of_Travel, Count_Encoding_Class, Count_Encoding_Flight_Distance_Binned_10, Count_Encoding_CustomerType_TypeOfTravel, Count_Encoding_Class_TypeOfTravel |
| 11 | Deep Dive into Delay and Rating Interaction Features | CatBoost | Delay_Total, Delay_Diff, Delay_Ratio, Zero_Ratings_Count, Net_Satisfaction_Score, NonZero_Rating_Mean, NonZero_Rating_Std, Wifi_x_Boarding, Comfort_x_Clean, Entertainment_x_Wifi, Wifi_is_zero, Boarding_is_zero, Clean_Food_Ratio, Wifi_Booking_Gap, Digital_Service_Index, Digital_Satisfaction_Gap, Physical_Service_Index, Cabin_Comfort_Cluster, Service_Expectation_Gap, Wifi_Boarding_Ratio, Loyal_Business_Segment |
| 12 | Route-based Aggregations with Binned Flight Distance | LightGBM | Flight_Distance_Binned_10, Route_Mean_Wifi_Service_BinnedFD, Route_Mean_Seat_Comfort_BinnedFD, Route_Mean_Arrival_Delay_BinnedFD, Route_Std_Arrival_Delay_BinnedFD, Relative_Wifi_Route_BinnedFD, Relative_Seat_Route_BinnedFD, Relative_Delay_Route_BinnedFD |
| 13 | External Data-Enhanced Target Encoding for Route-based Features | LightGBM | TE_External_FlightDistance_Class, TE_External_FlightDistance_TypeOfTravel, TE_External_CustomerType_TypeOfTravel |
| 14 | Advanced Interaction Features from Public Notebooks | XGBoost | Delay_Per_Mile, Zero_Ratings_Count, Net_Satisfaction_Score, NonZero_Rating_Mean, NonZero_Rating_Std, NonZero_Ratings_Ratio, Count_Top_Ratings, Count_Low_Ratings, Wifi_x_Boarding, Comfort_x_Clean, Entertainment_x_Wifi, Wifi_is_zero, Boarding_is_zero, Clean_Food_Ratio, Wifi_Booking_Gap, Digital_Service_Index, Digital_Satisfaction_Gap, Physical_Service_Index, Cabin_Comfort_Cluster, Service_Expectation_Gap, Wifi_Boarding_Ratio, Loyal_Business_Segment |
| 15 | Refined Route-based Aggregations with External Data | CatBoost | Flight_Distance_Binned_10, External_Route_Mean_Wifi_Service, External_Route_Mean_Seat_Comfort, External_Route_Mean_Arrival_Delay, External_Route_Std_Arrival_Delay, Relative_Wifi_External_Route, Relative_Seat_External_Route, Relative_Delay_External_Route |
| 16 | Ablation: Remove Low-Performing CatBoost Features | CatBoost | Mean_Service_Rating, Delay_Total, Delay_Ratio |

## Guidance for the researcher
- The 'id' column must be excluded from training features.
- All feature engineering steps must be performed in a fold-safe manner (e.g., target encoding should be done within CV folds to prevent leakage).
- For 'Arrival Delay in Minutes' missing values, use mean imputation or a constant (e.g., 0 or -1) as a first approach. The public notebooks suggest filling with 'Departure Delay in Minutes' or 0.
- The target variable 'satisfaction' should be mapped to 0 and 1 for binary classification.
- Ensure consistent random seeds across all experiments for reproducibility and fair paired comparisons.
- Optimize directly for ROC AUC score.
- For 'Arrival Delay in Minutes' missing values, prioritize filling with 'Departure Delay in Minutes' if available, then 0.0, as suggested by public notebooks. This overrides earlier guidance to use mean imputation or a constant (e.g., -1).
- When creating count or target encodings, ensure they are implemented within the `build_model` pipeline using scikit-learn transformers (e.g., `TargetEncoder`, `CountEncoder` if available or custom) to guarantee fold-safe cross-fitting and prevent data leakage.
- When creating binned 'Flight Distance' features, use integer division by 10 (e.g., `df['Flight Distance'] // 10`) as suggested by public notebooks.
- For all count and target encodings, ensure they are implemented as scikit-learn compatible transformers within the `build_model` pipeline to guarantee fold-safe cross-fitting and prevent data leakage. Do not perform these in `make_features`.
- When using external data for feature engineering (e.g., target encoding, global aggregations), ensure the external data is loaded and pre-processed *once* outside the CV loop. The derived mappings or statistics should then be applied to the internal training and validation sets within each CV fold to prevent leakage and ensure fold-safety. The external data should be treated as a 'prior' or 'global' knowledge source.
- For XGBoost, explicitly ensure that boolean features are converted to integers (0 or 1) before being passed to the model to avoid `ValueError: 'True' is not in list` issues. This conversion should happen within the `build_model` pipeline or `make_features` if stateless.

