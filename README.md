# CSI5388-Project

WIP

dataset.zip
  day1.csv contains ~56000 sample from each attack type (including benign) from day 1.
  day2.csv contains ~56000 sample from each attack type (including benign) from day 2.

binary_multiclass.zip
  binary.csv contains ~56000 benign cases from Day 2 and ~56000 attack cases from Day 1 (equal proportions of each attack)
  multiclass.csv contains ~56000 samples of each attack from Day 1.
  
assembler.ipynb is the script used to prepare day1.csv and day2.csv. Uses the Dask library to be able to process large datasets.
dataset_bin_mul.ipynb is used to convert the above's output into datasets suitable for binary and multiclass classification.



testing scripts folder has a few test scripts used on the old dataset. Uploaded as an example of an ML pipeline used on this dataset.
