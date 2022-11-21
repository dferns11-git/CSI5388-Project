# CSI5388-Project

WIP

binary_multiclass_extracted_cleaned.zip
  binary.csv contains ~54000 benign cases from Day 2 and ~54000 attack cases from Day 1 (equal proportions of each attack). Attacks not binarized.
  multiclass.csv contains ~54000 samples of each attack from Day 1.
  
extract_and_clean.ipynb is the script used to prepare and clean the original dataset files. Uses the Dask library to be able to process large datasets.

testing scripts folder has some examples of feature selection. Also contains scripts run on an old extract to show examples of ML pipeline.
