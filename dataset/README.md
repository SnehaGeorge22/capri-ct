# Data Preprocessing 

Data preprocessing is the process of transforming raw data into a clean and structured format that can be effectively used for analysis or machine learning tasks. It is a critical step in any data science or medical imaging pipeline, especially in high-stakes domains like healthcare and radiology, where data quality directly affects model reliability.

Points to remember before working with the dataset
--------------------------------------------------
- Unzip the img folder inside dataset folder to get all the CT images in /dataset/img/ directory
- "original dataset.xlsx" is the initial version of the dataset received without preprocessing (At this point , we could see that the dataset contains the file location with the amplification details available in the same column. Hence , we are going to perform VLOOKUP operation in the excel file and then separate the image filename, Voltage and Current per second(Time) from the original dataset so that it is in the format for ML model consumption )
- "processed_dataset.csv" is the second version of the dataset with preprocessing done.
- "final_dataset.csv" is the final version of the dataset after filtering the data based on SNR distribution

Note:
-----
Refer Data_preprocessing.ipynb inside notebooks folder for more details 
