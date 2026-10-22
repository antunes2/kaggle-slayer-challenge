# Data report — binary_classification on `satisfaction`

This dataset contains airline passenger information with the goal of predicting customer satisfaction. The target variable 'satisfaction' is binary, with a slight imbalance (55.6% False, 44.4% True). The evaluation metric is ROC AUC, which is suitable for imbalanced binary classification.

## Key findings
- The target variable 'satisfaction' has a class distribution of 55.6% False and 44.4% True, indicating a mild imbalance.
- The evaluation metric is ROC AUC, which is robust to class imbalance and appropriate for this binary classification task.
- Several features show strong positive correlations with the target, including 'Online boarding' (0.5596), 'Inflight entertainment' (0.4308), and 'Seat comfort' (0.4015).
- All features appear to be numerical (int64 or float64) or categorical with low cardinality (max 6 unique values for service ratings, 2-3 for customer/travel/class types).
- The 'id' column is a unique identifier and should not be used as a feature.
- The 'Arrival Delay in Minutes' column has a small percentage of missing values (0.04%).

## Data quality issues
- The 'id' column is an identifier and should be excluded from training.
- The 'Arrival Delay in Minutes' column has a small percentage of missing values (0.04%) which may require imputation.

## Risks
- While the class imbalance is mild, extreme models could still be biased towards the majority class. ROC AUC is a good metric, but monitoring other metrics like Precision-Recall AUC might be beneficial.
- The profile does not provide information about potential train-test drift. Cross-validation strategies should be chosen carefully to mitigate this risk.
- The profile does not show correlations between features, so potential multicollinearity issues are not immediately apparent.
- The 'id' column, if accidentally included, could lead to data leakage or overfitting.

## Feature plan
| idea | expected | columns | rationale |
|---|---|---|---|
| Service_Rating_Interaction | medium | Inflight wifi service, Departure/Arrival time convenient, Ease of Online booking, Gate location, Food and drink, Online boarding, Seat comfort, Inflight entertainment, On-board service, Leg room service, Baggage handling, Checkin service, Cleanliness | Many service-related features (e.g., 'Inflight wifi service', 'Food and drink', 'Online boarding') have moderate correlations with satisfaction. Combining them, perhaps through averaging or weighted sums, could capture synergistic effects. |
| Age_Flight_Distance_Interaction | medium | Age, Flight Distance | Age and Flight Distance are numerical features. Their interaction might reveal different satisfaction patterns for different age groups on varying flight lengths. |
| Customer_Type_Travel_Class_Interaction | medium | Customer Type, Type of Travel, Class | Interactions between categorical features like 'Customer Type', 'Type of Travel', and 'Class' could reveal distinct satisfaction drivers for different customer segments. |
| Delay_Ratio | low | Arrival Delay in Minutes, Flight Distance | The ratio of Arrival Delay to Flight Distance could indicate the severity of delays relative to the expected travel time, potentially impacting satisfaction. |
| Service_Per_Category_Count | low | Inflight wifi service, Departure/Arrival time convenient, Ease of Online booking, Gate location, Food and drink, Online boarding, Seat comfort, Inflight entertainment, On-board service, Leg room service, Baggage handling, Checkin service, Cleanliness | Count the number of services rated above a certain threshold (e.g., 3 or 4) to capture overall service quality perception. |

