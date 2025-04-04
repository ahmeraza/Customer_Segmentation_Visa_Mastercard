# Customer Segmentation for Visa & Mastercard Holders

The dataset contains anonymized customer details including credit card usage, and financial 
attributes for Visa and Mastercard holders with their profie demographic factors 
(gender, race, age, etc.)

## Description

The project delivers actionable customer profiles for strategic decision-making 
and marketing personalization. To prepare the dataset for K-Means clustering, we applied comprehensive feature engineering 
techniques. Categorical variables like gender and education were converted to numerical 
values, while one-hot encoding was used for marital status due to its non-ordinal nature. 

This transformation ensured the data was machine-readable and retained its categorical 
distinctions. Next, we applied feature scaling using StandardScaler, a critical step 
because K-Means relies on Euclidean distance. Standardizing features ensured each variable 
contributed equally to the clustering process, preventing high-magnitude variables like 
income or credit limit from dominating the results. Once scaled, the dataset was 
subjected to K-Means clustering using the KMeans algorithm from scikit-learn. 
We determined the optimal number of clusters using the Elbow Method and conducted 
a detailed evaluation of each cluster’s composition. This included analyzing income, 
usage patterns, and demographic distribution. In the conclusion phase, we defined 
each cluster’s key characteristics and offered targeted insights, such as which 
groups are underutilizing their cards or may benefit from higher credit limits. 

We used logistic regression and random forest models to predict income. Logistic
regression struggles with class imbalance, while the random forest model achieves 85%
accuracy, with better precision and recall for high-income predictions (>50K). However,
its 61% recall for the >50K class indicates challenges in accurately identifying high
earners, particularly females, blacks, and other people from underrepresented groups.
The analysis highlights systemic income disparities tied to race and gender, reflecting
broader societal inequities.

Key tasks include:

- Data cleaning and preprocessing using pandas and numpy
- Exploratory data analysis and visualization using matplotlib and seaborn
- Feature scaling using StandardScaler
- K-Means Clustering for customer segmentation
- Visual representation of customer segments for business interpretati


## Outcome

- Cleaned and preprocessed real-world customer data
- Built a reliable clustering model using K-Means
- Delivered actionable business insights for each customer segment

## Installation
To set up this project locally:
1. Clone the repository:
   ```bash
   git clone git@github.com:ahmeraza/Customer_Segmentation_Visa_Mastercard.git
```