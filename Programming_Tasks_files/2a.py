import csv
dataPath = '/pathToMyData/'
from collections import Counter
brands = Counter()
with open(dataPath + 'programming-tasks/top10_sample.csv', 'r') as datafile:
	datareader = csv.reader(datafile)
	for row in datareader:
		for brand in row[0][1:-1].split(','):
			brands[brand] += 1
brands.most_common(len(brands))