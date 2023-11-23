# Classification_of_Pediatric_Psychiatric_Disorders

## Reference
These data are collected in National Taiwan University Hospital Hsin-Chu Branch.  
For the detail of this experiment, please read my master thesis.

---
## About
In this study, we employed non-invasive, non-radiation, and portable near-infrared spectroscopy to measure changes in prefrontal lobe oxygenation during the Posner cueing paradigm in patients with TS, ADHD, and comorbid conditions. Through oxygenation data and deep learning, we developed disease classification models, achieving training and testing accuracies of 82.1% and 78.9% in distinguishing healthy controls, TS, and ADHD, and 85.4% and 63.6% in distinguishing comorbid cases, TS, and ADHD in the training and testing phases, respectively. Furthermore, we applied near-infrared spectroscopy to conduct neurofeedback training in children with TS. We recorded their oxygenation signals during the Posner cueing paradigm in the first week and attempted to differentiate responders to this therapy, achieving 100\% training accuracy and 75\% testing accuracy. These results confirm the feasibility of using near-infrared spectroscopy combined with deep learning for assessing childhood mental disorders.

---
## Files
```
 Path
    ├ 1_data_preprocessing.ipynb
    └ 2_deep_learning.ipynb
```

## Data Preprocessing
Four steps is operated in this file
1. Caculating oxygen concentration of blood
2. Data cleaning
3. Filtering
4. Normalization

## Machine Learning
