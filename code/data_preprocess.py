"""
Data Preprocessor
=================
Usage: python data_preprocess.py -i INPUT_FILE_PATH -o OUTPUT_FILE_PATH -t TOKENIZER_DIR

Requires installation of the Stanford Tokenizer (https://nlp.stanford.edu/software/tokenizer.shtml)
and Java 1.8. (This may require running 'module load java'.)

Assumes the following default paths:
    DEFAULT_INPUT_PATH = '../data/quora/quora_duplicate_questions.tsv'
    DEFAULT_OUTPUT_PATH = '../data/quora/tokenized_data.tsv'
    DEFAULT_TOKENIZER_DIR = '../../stanford-corenlp-full-2016-10-31/'

The input file should be a TSV (tab-separated values) file with 6 columns:
    id      qid1    qid2    question1       question2       is_duplicate

The output file will have the following format where question1 and question2 tokenized
questions with tokens separated by spaces:
    question1
    question2
    label
    question1
    question2
    label
    ...
"""

from __future__ import print_function
import argparse
import csv
import subprocess
import os

DEFAULT_INPUT_PATH = '../data/quora/quora_duplicate_questions.tsv'
DEFAULT_OUTPUT_PATH = '../data/quora/tokenized_data.tsv'
DEFAULT_TOKENIZER_DIR = '../../stanford-corenlp-full-2016-10-31/'

def main(input_path, output_path, tokenizer_dir):
    print('input path: %s' % input_path)
    print('output path: %s' % output_path)
    print('tokenizer dir: %s' % tokenizer_dir)

    # convert input path to desired output format, write to temp file
    temp_path = output_path + '.tmp'
    print("Writing to temp file: %s" % temp_path)

    with open(input_path, 'r') as input_file, open(temp_path, 'w') as temp_file:
        reader = csv.reader(input_file, delimiter='\t')

        isHeader = True
        firstLine = True
        for line in reader:

            # bypass header
            if isHeader:
                isHeader = False
                continue

            # trim whitespace from beginning and end of each line,
            # then merge all whitespace runs into 1 whitepace
            sent1 = ' '.join(line[3].split())
            sent2 = ' '.join(line[4].split())
            label = line[5].strip()

            if firstLine:
                temp_file.write(sent1 + '\n')
                firstLine = False
            else:
                temp_file.write('\n' + sent1 + '\n')
            
            temp_file.write(sent2 + '\n')
            temp_file.write(label)

    # build the Stanford Tokenizer command. Example:
    #   java -cp "*" edu.stanford.nlp.process.PTBTokenizer -preserveLines -lowerCase -options "ptb3Escaping=false,normalizeOtherBrackets=false,normalizeParentheses=false" < ../cs224n-project/data/quora/tokenized_data.tsv > tokenized_data.tsv.out
    command = 'java -cp "%s" edu.stanford.nlp.process.PTBTokenizer' % os.path.join(tokenizer_dir, '*')
    command += ' -preserveLines -lowerCase -options "ptb3Escaping=false,normalizeOtherBrackets=false,normalizeParentheses=false,tokenizePerLine=true"'
    command += ' < %s > %s' % (temp_path, output_path)

    print("Running the tokenizer with the following command:")
    print("\t%s" % command)

    # tokenize the output file, and remove the temp file
    subprocess.call(command, shell=True)
    os.remove(temp_path)

if __name__ == "__main__":
    description = "Tokenize the Quora dataset"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-i", "--input", required=False, help="path to input Quora dataset file (quora_duplicate_questions.tsv)")
    parser.add_argument("-o", "--output", required=False, help="path to output tokenized file")
    parser.add_argument("-t", "--tokenizer", required=False, help="path to Stanford Tokenizer directory")
    args = parser.parse_args()

    input_path = DEFAULT_INPUT_PATH
    if args.input is not None:
        input_path = args.input

    output_path = DEFAULT_OUTPUT_PATH
    if args.output is not None:
        output_path = args.output

    tokenizer_dir = DEFAULT_TOKENIZER_DIR
    if args.tokenizer is not None:
        tokenizer_dir = args.tokenizer
    
    main(input_path, output_path, tokenizer_dir)