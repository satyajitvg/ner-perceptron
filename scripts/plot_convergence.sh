# driver script to plot number of iterations & model accuracy
# $1 train data fpath
# $2 test data fpath

for num_iter in 1 3 5 10 15 20 25; do
    python train_ner.py -trainfile $1 -testfile $2 -iter $num_iter -cf n
done
