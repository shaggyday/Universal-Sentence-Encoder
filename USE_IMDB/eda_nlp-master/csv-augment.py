# Easy data augmentation techniques for text classification
# Jason Wei and Kai Zou

from eda import *
import pandas as pd
#arguments to be parsed from command line
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True, type=str, help="input file of unaugmented data")
ap.add_argument("--output", required=False, type=str, help="output file of unaugmented data")
ap.add_argument("--num_aug", required=False, type=int, help="number of augmented sentences per original sentence")
ap.add_argument("--alpha", required=False, type=float, help="percent of words in each sentence to be changed")
args = ap.parse_args()

#the output file
output = None
if args.output:
    output = args.output
else:
    from os.path import dirname, basename, join
    output = join(dirname(args.input), 'eda_' + basename(args.input))

#number of augmented sentences to generate per original sentence
num_aug = 9 #default
if args.num_aug:
    num_aug = args.num_aug

#how much to change each sentence
alpha = 0.001 #default
if args.alpha:
    alpha = args.alpha

#generate more data with standard augmentation
def gen_eda(train_orig, output_file, alpha, num_aug=9):

    # writer = open(output_file, 'w')
    # lines = open(train_orig, 'r').readlines()

    df = pd.read_csv('movie_reviews_review_level.csv')
    # df_aug = pd.DataFrame()
    # df_aug['review'] = ''
    # df_aug['sentiment'] = ''
    aug_data = []

    for i in range(len(df.index)):
        print(i)
        cur_row = df.loc[i]
        label = cur_row['sentiment']
        sentence = cur_row['review']
        aug_sentences = eda(sentence, alpha_sr=alpha, alpha_ri=alpha, alpha_rs=alpha, p_rd=alpha, num_aug=num_aug)
        for aug_sentence in aug_sentences:
            # print(aug_sentence)
            dict1 = {'review': aug_sentence, 'sentiment': label}
            # new_row = pd.Series([aug_sentence, label], index = ['review', ['sentiment']])
            # df_aug = df_aug.append(new_row, ignore_index=True)
            aug_data.append(dict1)

    df_aug = pd.DataFrame(aug_data)
    file_name = "IMDB_num_aug=" + str(num_aug) + ",alpha=" + str(alpha) + ".csv"
    df_aug.to_csv(file_name)
    # print("generated augmented sentences with eda for " + train_orig + " to " + output_file + " with num_aug=" + str(num_aug))

#main function
if __name__ == "__main__":

    #generate augmented sentences and output into a new file
    gen_eda(args.input, output, alpha=alpha, num_aug=num_aug)