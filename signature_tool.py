#!/usr/bin/python

import sys
import pickle
import signature_read
import signature_make

def main():
    action = sys.argv[1]

    if action == "compact":
        level_filename = sys.argv[2]
        feature_filename = sys.argv[3]
        result = make_signature(level_filename, feature_filename)

    if action == "print":
        level_filename = sys.argv[2]
        feature_filename = sys.argv[3]
        result = make_signature(level_filename, feature_filename)
        for index, value in result:
            print "%d %s" % (index / 100, bin(value))
        for value in result:
            print value

    if action == "compare":
        first_level_filename = sys.argv[2]
        first_feature_filename = sys.argv[3]
        second_level_filename = sys.argv[4]
        second_feature_filename = sys.argv[5]
        result = compare_signatures(first_level_filename, first_feature_filename, second_level_filename, second_feature_filename)
        print result

    if action == "index":
        level_filelist = sys.argv[2]
        feature_filelist = sys.argv[3]
        level_filenames = [v.strip() for v in open(level_filelist, "r").readlines()]
        feature_filenames = [v.strip() for v in open(feature_filelist, "r").readlines()]
        data = index_signatures(level_filenames, feature_filenames)
        print pickle.dumps(data)

    if action == "search":
        indexname = sys.argv[2]
        level_filelist = sys.argv[3]
        feature_filelist = sys.argv[4]

        level_filenames = [v.strip() for v in open(level_filelist, "r").readlines()]
        feature_filenames = [v.strip() for v in open(feature_filelist, "r").readlines()]
        result = search_signature(indexname, level_filename, feature_filename)
        print level_filename
        for search_i, search_compare in result:
            print "\t%s = %f" % (level_filenames[search_i], search_compare)

    if action == "search_all":
        indexname = sys.argv[2]
        level_filelist = sys.argv[3]
        feature_filelist = sys.argv[4]
        threshold = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0
        level_filenames = [v.strip() for v in open(level_filelist, "r").readlines()]
        feature_filenames = [v.strip() for v in open(feature_filelist, "r").readlines()]
        result = search_signatures(indexname, level_filenames, feature_filenames)
        for i, search_result in enumerate(result):
            print level_filenames[i]
            for search_i, search_compare in search_result:
                if search_compare >= threshold:
                    print "\t%s = %f (%f)" % (level_filenames[search_i], search_compare, 0)

    if action == "index_level":
        import normalize
        level_filelist = sys.argv[2]

        level_filenames = [v.strip() for v in open(level_filelist, "r").readlines()]
        signature_level_index = index_level_signatures(level_filenames)
        print pickle.dumps(signature_level_index)

def index_level_signatures(level_filenames):
    signature_matrixes = []
    for level_filename in level_filenames:
        signature = signature_read.read_signature_short(level_filename)
        signature = normalize.wline_normalize(signature)
        signature = [int(v) for v in normalize.line_normalize(signature, -2**15, 2**15 - 1)]
        signature_matrix = signature_make.signature_matrix(signature)
        signature_matrixes.append(signature_matrix)
    return signature_matrixes

def compare_signatures(first_level_filename, first_feature_filename, second_level_filename, second_feature_filename):
    first_level_signature = signature_read.read_signature_short(first_level_filename)
    first_feature_signature = signature_read.read_signature_byte(first_feature_filename)
    second_level_signature = signature_read.read_signature_short(second_level_filename)
    second_feature_signature = signature_read.read_signature_byte(second_feature_filename)
    first_signature = signature_make.signature_make(first_level_signature, first_feature_signature)
    second_signature = signature_make.signature_make(second_level_signature, second_feature_signature)

    first_signature.sort()
    second_signature.sort()

    return signature_make.signature_compare(first_signature, second_signature)

def make_signature(level_filename, feature_filename):
    level_signature = signature_read.read_signature_short(level_filename)
    feature_signature = signature_read.read_signature_byte(feature_filename)
    return signature_make.signature_make(level_signature, feature_signature)

def index_signatures(level_filenames, feature_filenames):
    signatures = []
    for level_filename, feature_filename in zip(level_filenames, feature_filenames):
        level_signature = signature_read.read_signature_short(level_filename)
        feature_signature = signature_read.read_signature_byte(feature_filename)
        signature = signature_make.signature_make(level_signature, feature_signature)
        signatures.append(signature)
    return signature_make.signature_index(signatures)

def search_signature(indexname, level_filename, feature_filename):
    data = pickle.loads(open(indexname, "r").read())

    search_result = signature_make.signature_search_intersect(data, signature)
    return search_result

def search_signatures(indexname, level_filenames, feature_filenames):
    import correlate
    import smooth_avg

    level_signatures = []
    for level_filename, feature_filename in zip(level_filenames, feature_filenames):
        level_signature = signature_read.read_signature_short(level_filename)
        level_signatures.append(level_signature)

    data = pickle.loads(open(indexname, "r").read())
    search_results = []
    for level_filename, feature_filename in zip(level_filenames, feature_filenames):
        level_signature = signature_read.read_signature_short(level_filename)
        feature_signature = signature_read.read_signature_byte(feature_filename)
        signature = signature_make.signature_make(level_signature, feature_signature)
        search_result = signature_make.signature_search_intersect(data, signature)
        search_result = [(i, compare, correlate.pearson(smooth_avg.avg_smooth(level_signature), smooth_avg.avg_smooth(level_signatures[i]))) for i, compare in search_result]
#        search_result = [(i, compare) for i, compare in search_result]
        search_results.append(search_result)
    return search_results

def adjacent(iterator, start = None):
    iterator = iter(iterator)
    lastvalue = start
    if not lastvalue:
        lastvalue = next(iterator, None)
    for value in iterator:
        yield (lastvalue, value)
        lastvalue = value

if __name__ == "__main__":
    main()
