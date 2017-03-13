from autocorrect import spell
import csv
with open('../data/quora/train.tsv') as f, open('../data/quora/spellchecked/train_sp.tsv', 'w+') as w:
    reader = csv.reader(f, delimiter='\t')
    writer = csv.writer(w, delimiter='\t')
    count = 0
    for line in reader:
        new_line = []
        for word in line:
            if len(word) > 3:
                new_line.append(spell(word).lower())
            else:
                new_line.append(word)
            
        writer.writerow(new_line)
        count += 1
        if count % 10000 == 0:
            print(count)
        
with open('../data/quora/test.tsv') as f, open('../data/quora/spellchecked/test_sp.tsv', 'w+') as w:
    reader = csv.reader(f, delimiter='\t')
    writer = csv.writer(w, delimiter='\t')
    count = 0
    for line in reader:
        new_line = []
        for word in line:
            if len(word) > 3:
                new_line.append(spell(word).lower())
            else:
                new_line.append(word)
            
        writer.writerow(new_line)
        count += 1
        if count % 10000 == 0:
            print(count)