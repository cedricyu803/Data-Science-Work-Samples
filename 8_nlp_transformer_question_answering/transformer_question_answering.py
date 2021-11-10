# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 18:00:00 2021

@author: Cedric Yu
"""

"""

# Extractive Question Answering
# Question answering (QA) is a task of natural language processing that aims to automatically answer questions. 
# The goal of [extractive QA] is to identify [the portion of the text] that contains the (given) answer to a question. 
    # For example, when tasked with answering the question 'When will Jane go to Africa?' given the text data 'Jane visits Africa in September', the question answering model will hightlight 'September' in the sentence by finding the start and end string indices of the 'September'

# We will use a variation of the Transformer model to answer questions about stories, and fine-tune it to a custom dataset
# We will implement extractive QA model in TensorFlow (and in PyTorch).

--------------------------


"""

#%% Preamble

import os
os.chdir(r'C:\Users\Cedric Yu\Desktop\Data Science\Works\11_nlp_transformer_question_answering')

import pandas as pd
import numpy as np


#%% tensorflow

# import tensorflow as tf
# from keras.models import Model
# from keras.layers import Input, Dense, Reshape, merge
# from keras.layers.embeddings import Embedding
# from keras.preprocessing.sequence import skipgrams
# from keras.preprocessing import sequence

# from tensorflow.random import set_seed
# set_seed(0)
# np.random.seed(1)


#%% the QA bAbI dataset

"""
We are given one of the bAbI datasets generated by Facebook AI Research
"""

from datasets import load_from_disk

# Load a dataset and print the first example in the training set
babi_dataset = load_from_disk('datasets/')
# DatasetDict({
#     train: Dataset({
#         features: ['story'],
#         num_rows: 1000
#     })
#     test: Dataset({
#         features: ['story'],
#         num_rows: 1000
#     })
# })

"""
For a given story, there are two sentences which serve as the [context], and one [question]. Each of these phrases has an [ID]. There is also a [supporting fact ID] which refers to a sentence in the story that helps answer the question. 

# For example, for the question 'What is east of the hallway?', the supporting fact 'The bedroom is east of the hallway' has the ID '2'. There is also the answer, 'bedroom' for the question.
"""

babi_dataset['train'][102]
# {'story': {'answer': ['', '', 'bedroom'],
#   'id': ['1', '2', '3'],
#   'supporting_ids': [[], [], ['2']],
#   'text': ['The bedroom is west of the office.',
#    'The bedroom is east of the hallway.',
#    'What is east of the hallway?'],
  # 'type': [0, 0, 1]}}



""" 
All stories in the dataset are of the same format: two context sentences and one question
"""

type_set = set()
for story in babi_dataset['train']:
    if str(story['story']['type'] )not in type_set:
        type_set.add(str(story['story']['type'] ))
# type_set
# Out[12]: {'[0, 0, 1]'}


"""
# flatten the dataset for easier manipulation
# add dictionary items for simpler output
"""

flattened_babi = babi_dataset.flatten()

def get_question_and_facts(story):
    dic = {}
    dic['question'] = story['story.text'][2]
    # join the two sentences by a whitespace
    dic['sentences'] = ' '.join([story['story.text'][0], story['story.text'][1]])
    dic['answer'] = story['story.answer'][2]
    return dic

processed = flattened_babi.map(get_question_and_facts)

# processed['train'][0]
# {'story.answer': ['', '', 'office'],
#  'story.id': ['1', '2', '3'],
#  'story.supporting_ids': [[], [], ['1']],
#  'story.text': ['The office is north of the kitchen.',
#   'The garden is south of the kitchen.',
#   'What is north of the kitchen?'],
#  'story.type': [0, 0, 1],
#  'question': 'What is north of the kitchen?',
#  'sentences': 'The office is north of the kitchen. The garden is south of the kitchen.',
#  'answer': 'office'}

# processed['test'][2]
# {'story.answer': ['', '', 'bathroom'],
#  'story.id': ['1', '2', '3'],
#  'story.supporting_ids': [[], [], ['1']],
#  'story.text': ['The bathroom is north of the garden.',
#   'The hallway is north of the bathroom.',
#   'What is north of the garden?'],
#  'story.type': [0, 0, 1],
#  'question': 'What is north of the garden?',
#  'sentences': 'The bathroom is north of the garden. The hallway is north of the bathroom.',
#  'answer': 'bathroom'}

"""
get the start and end indices of the answer in 'sentences' for each of the stories
"""

def get_start_end_idx(story):
    str_idx = story['sentences'].find(story['answer'])
    end_idx = str_idx + len(story['answer'])
    return {'str_idx':str_idx,
          'end_idx': end_idx}

processed = processed.map(get_start_end_idx)

# processed['train'][78]
# {'story.answer': ['', '', 'office'],
#  'story.id': ['1', '2', '3'],
#  'story.supporting_ids': [[], [], ['1']],
#  'story.text': ['The office is north of the bathroom.',
#   'The kitchen is south of the bathroom.',
#   'What is north of the bathroom?'],
#  'story.type': [0, 0, 1],
#  'question': 'What is north of the bathroom?',
#  'sentences': 'The office is north of the bathroom. The kitchen is south of the bathroom.',
#  'answer': 'office',
#  'str_idx': 4,
#  'end_idx': 10}


#%% tokenisation and label (tag id) alignment with the huggingface library

"""
token-label alignment:
same procedure as in the named-entity recognition
"""

from transformers import DistilBertTokenizerFast #, TFDistilBertModel
tokenizer = DistilBertTokenizerFast.from_pretrained('pre-trained-transformer-distilbert-base-cased/')

max_len = 512
def tokenize_align(example):
    
    # tokenize example['sentences'] + example['question'] into input_ids representing sub-words, with only one <start of sentence> token in front
    encoding = tokenizer(example['sentences'], example['question'], truncation=True, padding=True, max_length=max_len)
    
    # find out which token (sub-word) the first and last characters of the answer belong to
    start_positions = encoding.char_to_token(example['str_idx'])
    end_positions = encoding.char_to_token(example['end_idx']-1)
    if start_positions is None:
        start_positions = max_len
    if end_positions is None:
        end_positions = max_len
    return {'input_ids': encoding['input_ids'],
          'attention_mask': encoding['attention_mask'],
          'start_positions': start_positions,
          'end_positions': end_positions}

""" keep only the processed items"""

qa_dataset = processed.map(tokenize_align)
qa_dataset = qa_dataset.remove_columns(['story.answer', 'story.id', 'story.supporting_ids', 'story.text', 'story.type'])


""" Example"""
# ex0 = processed['train'][99]
# # np.array(tokenizer.tokenize(ex0['sentences'], ex0['question'], truncation=True, padding=True, max_length=tokenizer.model_max_length))
# # array(['the', 'kitchen', 'is', 'south', 'of', 'the', 'bathroom', '.',
#        # 'the', 'bedroom', 'is', 'north', 'of', 'the', 'bathroom', '.',
#        # 'what', 'is', 'the', 'bathroom', 'south', 'of', '?'], dtype='<U8')
# encoding0 = tokenizer(ex0['sentences'], ex0['question'], truncation=True, padding=True, max_length=tokenizer.model_max_length)
# # find out which token (sub-word) the first and last characters of the answer belong to
# start_positions0 = encoding0.char_to_token(ex0['str_idx'])
# # 10
# end_positions0 = encoding0.char_to_token(ex0['end_idx']-1)
# # 10


#%% model training

from transformers import TFDistilBertForQuestionAnswering
model = TFDistilBertForQuestionAnswering.from_pretrained("pre-trained-transformer-distilbert-base-cased", return_dict=False)

import tensorflow as tf

train_ds = qa_dataset['train']
test_ds = qa_dataset['test']
batch_size_train = len(train_ds)
batch_size_test = len(test_ds)

columns_to_return = ['input_ids','attention_mask', 'start_positions', 'end_positions']

"""
In the TensorFlow implementation, we have to set the data format type to tensors, which may create ragged tensors (tensors of different lengths).
We have to convert the ragged tensors to normal tensors using the to_tensor() method, which pads the tensors and sets the dimensions to [None, tokenizer.model_max_length], so we can feed different size tensors into a model based on the batch size.
"""

# train_ds.set_format(type='tf', columns=columns_to_return)

# dictionary of 'input_ids' and 'attention_mask', each is a tf tensor of shape (batch_size, max_len)

train_ds.set_format(type='tf', columns=columns_to_return)
train_features = {x: train_ds[x].to_tensor(default_value=0, shape=[None, max_len]) for x in ['input_ids', 'attention_mask']}

# each is a tf tensor of shape (batch_size, 1)
train_labels = {"start_positions": tf.reshape(train_ds['start_positions'], shape=[-1,1]),
                'end_positions': tf.reshape(train_ds['end_positions'], shape=[-1,1])}

train_tfdataset = tf.data.Dataset.from_tensor_slices((train_features, train_labels)).batch(2)


"""
fit the model with custom loss function, monitor process
# Target two loss functions, one for the start index and one for the end index.
# Create a custom training function using tf.GradientTape()
# tf.GradientTape() records the operations performed during forward prop for automatic differentiation during backprop.
"""

EPOCHS = 3
loss_fn1 = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn2 = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
opt = tf.keras.optimizers.Adam(learning_rate=3e-5)

losses = []
for epoch in range(EPOCHS):
    print("Starting epoch: %d"% epoch )
    for step, (x_batch_train, y_batch_train) in enumerate(train_tfdataset):
        with tf.GradientTape() as tape:
            answer_start_scores, answer_end_scores = model(x_batch_train)
            loss_start = loss_fn1(y_batch_train['start_positions'], answer_start_scores)
            loss_end = loss_fn2(y_batch_train['end_positions'], answer_end_scores)
            loss = 0.5 * (loss_start + loss_end)
        losses.append(loss)
        grads = tape.gradient(loss, model.trainable_weights)
        opt.apply_gradients(zip(grads, model.trainable_weights))

        if step % 20 == 0:
            print("Training loss (for one batch) at step %d: %.4f"% (step, 
                                                                   float(loss_start)))

# from matplotlib.pyplot import plot
# plot(losses)


#%% predictions

question, text = 'What is south of the bedroom?','The hallway is south of the garden. The garden is south of the bedroom.'

np.array(tokenizer.tokenize(text, question, truncation=True, padding=True, max_length=max_len))
# array(['the', 'hallway', 'is', 'south', 'of', 'the', 'garden', '.', 'the',
#        'garden', 'is', 'south', 'of', 'the', 'bedroom', '.', 'what', 'is',
#        'south', 'of', 'the', 'bedroom', '?'], dtype='<U7')

input_dict = tokenizer(text, question, return_tensors='tf')
# input_dict: tokenised 'input_ids' has shape (1,26)
outputs = model(input_dict)
# outputs is a tuple (start_logits, end_logits)
# each is the logits predicting the token (26 of them in this example) to which the start/end index of the answer belongs
start_logits = outputs[0]
end_logits = outputs[1]

# convert token ids in text + question to tokens
all_tokens = tokenizer.convert_ids_to_tokens(input_dict["input_ids"].numpy()[0])
# (['[CLS]', 'the', 'hallway', 'is', 'south', 'of', 'the', 'garden',
#        '.', 'the', 'garden', 'is', 'south', 'of', 'the', 'bedroom', '.',
#        '[SEP]', 'what', 'is', 'south', 'of', 'the', 'bedroom', '?',
#        '[SEP]']
answer = ' '.join(all_tokens[tf.math.argmax(start_logits, 1)[0] : tf.math.argmax(end_logits, 1)[0]+1])
print(question, answer.capitalize())
# What is south of the bedroom? Garden












