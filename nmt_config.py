# coding: utf-8

EXP_NAME = "ja_en_exp6"
NUM_SENTENCES = 10500
USE_ALL_DATA = True
FREQ_THRESH = 1
if USE_ALL_DATA:
  NUM_TRAINING_SENTENCES = NUM_SENTENCES-500
  NUM_DEV_SENTENCES = 500
else:
  NUM_TRAINING_SENTENCES = 200
  NUM_DEV_SENTENCES = 10
num_layers_enc = 1
num_layers_dec = 1
hidden_units = 100
use_attn = 2
load_existing_model = True
NUM_EPOCHS = 0
gpuid = -1


####################################################################################################
import os
input_dir = os.path.join("data")
model_dir = os.path.join("model")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(input_dir):
    print("Input folder not found".format(input_dir))
text_fname = {"en": os.path.join(input_dir, "text.en"), "fr": os.path.join(input_dir, "text.fr")}
tokens_fname = os.path.join(input_dir, "tokens.list")
vocab_path = os.path.join(input_dir, "vocab.dict")
w2i_path = os.path.join(input_dir, "w2i.dict")
i2w_path = os.path.join(input_dir, "i2w.dict")

max_vocab_size = {"en" : 10000, "fr" : 10000}

PAD = b"_PAD"
GO = b"_GO"
EOS = b"_EOS"
UNK = b"_UNK"
START_VOCAB = [PAD, GO, EOS, UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

NO_ATTN = 0
SOFT_ATTN = 1
HARD_ATTN = 2
attn_post = ["NO_ATTN", "SOFT_ATTN", "HARD_ATTN"]

print("Japanese English dataset configuration")
MAX_PREDICT_LEN = 20
name_to_log = "{0:d}sen_{1:d}-{2:d}layers_{3:d}units_{4:s}_{5:s}".format(
                                                            NUM_TRAINING_SENTENCES,
                                                            num_layers_enc,
                                                            num_layers_dec,
                                                            hidden_units,
                                                            EXP_NAME,
                                                            attn_post[use_attn])
log_train_fil_name = os.path.join(model_dir, "train_{0:s}.log".format(name_to_log))
model_fil = os.path.join(model_dir, "seq2seq_{0:s}.model".format(name_to_log))