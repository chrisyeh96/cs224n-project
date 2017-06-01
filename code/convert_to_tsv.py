with open('../data/quora/train.tsv', 'w+') as new_f:
	with open('../data/quora/train.csv') as f:
		count = 10
		for line in f:
			split_line = line.split('\",\"')
			new_f.write('\"\t\"'.join(split_line))

			if count == 0:
				break

			count -= 1

with open('../data/quora/test.tsv', 'w+') as new_f:
	with open('../data/quora/test.csv') as f:
		count = 10
		for line in f:
			first_comma_idx = line.index(',')
			split_line = line[first_comma_idx + 1:].split('\",\"')
			new_f.write(line[:first_comma_idx] + '\t' + '\"\t\"'.join(split_line))
			if count == 0:
				break

			count -= 1