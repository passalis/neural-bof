from dataset import get_15scene_splits, get_15scene_features
from model import NBoF

# Before running set the dataset path of the 15-scene dataset (download from http://www-cvr.ai.uiuc.edu/ponce_grp/data)
# and download the pretrained CNN used for extracting the features from https://github.com/metalbubble/places365

n_pretrain_iters = 5
n_train_iters = 30

# Get sample data (15-scene dataset) and get a random split
data, labels = get_15scene_features()
train_idx, test_idx = get_15scene_splits(seed=1)

nbow = NBoF(n_codewords=16, eta=0.01, eta_V=0.001, eta_W=0.2, g=0.01,
            update_V=True, update_W=True, n_hidden=100, n_output=15)

# Sample some feature vectors for initializing the dictionary
idx = sorted(train_idx)
nbow.bow.initialize_dictionary(data[idx, :, :], n_samples=5000)

# Pre-training
for i in range(n_pretrain_iters):
    nbow.do_train_epoch(train_idx, data, labels, batch_size=50, type='finetune')

(train_acc, train_prec) = nbow.evaluate_model(train_idx, data, labels)
(test_acc, test_prec) = nbow.evaluate_model(test_idx, data, labels)

print "Pre-train: training set mean precision = ", train_prec, " %"
print "Pre-train: testing set mean precision = ", test_prec, " %"

# Full-training
for i in range(n_train_iters):
    nbow.do_train_epoch(train_idx, data, labels, batch_size=50)

(train_acc, train_prec) = nbow.evaluate_model(train_idx, data, labels)
(test_acc, test_prec) = nbow.evaluate_model(test_idx, data, labels)

print "Full training: training set mean precision = ", train_prec, " %"
print "Full training: testing set mean precision = ", test_prec, " %"

