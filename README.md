# Classification_of_Pediatric_Psychiatric_Disorders

## Reference
These data are collected in National Taiwan University Hospital Hsin-Chu Branch.  
For the detail of this experiment, please read my master thesis.

---
## Content
1. About
2. Data Information 
3. Code Information
    1. Data Preprocess
    2. Feature Extraction
    3. Feature Statistic
    4. Deep Learning

---
## About
- In this study, 

## Data Information

### Data Information of Patients
The anonymous patient list is saved in `patient_list.csv`

| Column name | Details |
| --- | --- |
| Case | Case number|
| Gruop\* | control / ADHD / TS / both |

\* Diagnosed by a doctor. 

### Data information for NIRS Files
The files are placed by the case number, `usually` two files for each case. Check it by yourself.

\# Useful columns

| Column name | Details |
| --- | --- |
| Time_Host | Time |
| CH2_PD730 | The intensity in ch2 at 730nm, range in 0 to 4095 |
| CH2_PD850 | The intensity in ch2 at 850nm, range in 0 to 4095 |
| CH3_PD730 | The intensity in ch3 at 730nm, range in 0 to 4095 |
| CH3_PD850 | The intensity in ch3 at 850nm, range in 0 to 4095 |
| trail_times | Number of trails, 32 in total |

---

## Code Information
1. Data Preprocess
        
2. Feature Extraction
    - stage activation 


3. Feature Statistic
4. Machine Learning
