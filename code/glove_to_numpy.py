import numpy as np
import argparse
import pickle
import os
from fnmatch import fnmatch
from numpy.lib.format import open_memmap

DEFAULT_FILE_PATH = "../data/glove/glove.6B.300d.txt"
MAX_ROWS = 70000 # limit our memory usage

def convert_to_numpy(glove_text_path, out_dir):
    """Read pretrained GloVe vectors"""
    wordVectors = None # array
    words_to_index = {}
    curr_index = 0
    dimensions = None

    filename_prefix = glove_text_path.split('/')[-1].split('.')[0:-1]
    filename_prefix = '.'.join(filename_prefix)

    with open(glove_text_path) as glove_text_file:
        for line in glove_text_file:
            line = line.strip()
            if not line:
                continue
            
            row = line.split()
            if wordVectors is None:
                dimensions = len(row)-1
                wordVectors = np.zeros((MAX_ROWS, dimensions), dtype=np.float32)
            else:
                assert len(row)-1 == dimensions, "wrong number of dimensions"
            
            token = row[0]
            word_vector = np.asarray([float(x) for x in row[1:]])
            wordVectors[curr_index % MAX_ROWS] = word_vector
            words_to_index[token] = curr_index

            if (curr_index+1) % MAX_ROWS == 0:
                print "Finished %d tokens" % (curr_index+1,)
                temp_filepath = os.path.join(out_dir, '%s_%d.npy' % (filename_prefix, curr_index / MAX_ROWS))
                np.save(temp_filepath, wordVectors)

            curr_index += 1

    # save partial last batch
    if wordVectors is not None and curr_index % MAX_ROWS != 0:
        print "Finished %d tokens" % (curr_index,)
        temp_filepath = os.path.join(out_dir, '%s_%d.npy' % (filename_prefix, curr_index / MAX_ROWS))
        np.save(temp_filepath, wordVectors[: curr_index % MAX_ROWS])

    # save wordVectors
    merge_ondisk(out_dir, filename_prefix)

    # save word_to_index
    pkl_path = os.path.join(out_dir, filename_prefix+'.pkl')
    pickle.dump( words_to_index, open( pkl_path, "wb" ) )

def merge_ondisk(out_dir, key):
    '''
    Read numpy arrays in a directory with filenames of the form
    key_*.npy and merge them into a file named key.npy
    '''
    # determine the shape
    shape = None
    for fn in os.listdir(out_dir):
        if fnmatch(fn, '%s_*.npy' % key):
            fn_path = os.path.join(out_dir, fn)
            with open(fn_path, 'r') as f:
                version = np.lib.npyio.format.read_magic(f)
                if version == (1, 0):
                    hdr = np.lib.npyio.format.read_array_header_1_0(f)
                elif version == (2, 0):
                    hdr = np.lib.npyio.format.read_array_header_2_0(f)
                arr_shape = hdr[0]

            if shape is None:
                # arr_shape is a tuple (immutable), so convert to list (mutable)
                shape = list(arr_shape)
            else:
                shape[0] += arr_shape[0]

    # pre-allocate merged memmapped array
    merged_fn = os.path.join(out_dir, '%s.npy' % key)
    fp = open_memmap(merged_fn, dtype='float32',
                     mode='w+', shape=tuple(shape))
    curr_idx = 0
    for fn in sorted(os.listdir(out_dir)):
        if fnmatch(fn, '%s_*.npy' % key):
            fn_path = os.path.join(out_dir, fn)
            arr = np.load(fn_path, mmap_mode='r')
            fp[curr_idx:curr_idx+arr.shape[0]] = arr[:]
            curr_idx += arr.shape[0]
            del arr
            os.remove(fn_path)
    fp.flush()

def verify_npy(glove_text_path, out_dir):
    filename_prefix = glove_text_path.split('/')[-1].split('.')[0:-1]
    filename_prefix = '.'.join(filename_prefix)
    npy_path = os.path.join(out_dir, filename_prefix+".npy")
    pkl_path = os.path.join(out_dir, filename_prefix+".pkl")

    wordVectors_npy = np.load(npy_path, mmap_mode='r')
    with open(pkl_path, 'rb') as pkl_file:
        tokens_to_index = pickle.load(pkl_file)

    with open(glove_text_path) as glove_text_file:
        for line in glove_text_file:
            line = line.strip()
            if not line:
                continue
            
            row = line.split()
            token = row[0]
            word_vector = np.asarray([float(x) for x in row[1:]], dtype=np.float32)
            index = tokens_to_index[token]
            assert np.array_equal(word_vector, wordVectors_npy[index])

if __name__ == '__main__':
    # parse arguments. see top of file for explanation of arguments
    description = "Convert glove vectors from text file to numpy array and pkl"
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-i", "--input", required=False, default=DEFAULT_FILE_PATH, help="path to text file of glove vectors")
    parser.add_argument("-odir", "--output_dir", required=False, default="./", help="directory to store output npy and pkl files")
    parser.add_argument("-v", "--verify", required=False, action="store_true", help="only verify that the npy and pkl files are correct")
    args = parser.parse_args()

    glove_text_path = args.input
    out_dir = args.output_dir

    if not args.verify: # covert glove vectors from text file to npy array and pkl dictionary
        convert_to_numpy(glove_text_path, out_dir)
    else: # verify that we created the npy and pkl files correctly
        verify_npy(glove_text_path, out_dir)