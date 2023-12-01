# ds5010-disease-detection
## A Disease Detection Toolkit in Python

*Software Requirements:*
- Python 3 and above
- PIP 22 and above
  
*Steps to Install:*
Open Terminal and Run: ```pip install disease-detection``` or to upgrade, run 
```pip install --upgrade disease-detection```
(this includes installation of all required packages like numpy, pandas, scikit-learn, etc. to run our main package)


### *For cancer classification testing:*
```ruby
# import package/module on your Python env:
from disease_detection import cancer

inp_arr = [[13.54,	14.36,	87.46,	566.3,	0.09779,	0.08129,	0.06664,	0.04781,	0.1885,	0.05766,	0.2699,	0.7886,	2.058,	23.56,	0.008462,	0.0146,	0.02387,	0.01315,	0.0198,	0.0023,	15.11,	19.26,	99.7,	711.2,	0.144,	0.1773,	0.239,	0.1288,	0.2977,	0.07259	],
[19.81,	22.15,	130,	1260,	0.09831,	0.1027,	0.1479,	0.09498,	0.1582,	0.05395,	0.7582,	1.017,	5.865,	112.4,	0.006494,	0.01893,	0.03391,	0.01521,	0.01356,	0.001997,	27.32,	30.88,	186.8,	2398,	0.1512,	0.315,	0.5372,	0.2388,	0.2768,	0.07615	]]

g = cancer.cancer(inp_arr)

print(g.LogisticRegression())
print(g.KNearestNeighbours())
print(g.SupportVectorClassifier())
print(g.GNB())
print(g.RandomForest())
```

Note that the length of the input array must be 30 for cancer, and all values should be numeric. Also, note that the input_array is a list of arrays - you can classify for any number of rows and our package should be able to handle this.
