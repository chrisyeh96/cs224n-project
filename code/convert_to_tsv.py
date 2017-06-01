with open('train.tsv', 'w+') as new_f:
	with open('train.csv') as f:
		count = 10
		for line in f:
			split_line = line.split('\",\"')
			new_f.write('\"\t\"'.join(split_line))

with open('test.tsv', 'w+') as new_f:
	with open('test.csv') as f:
		count = 10
		for line in f:
			split_line = line.split('\",\"')
			new_f.write('\"\t\"'.join(split_line))