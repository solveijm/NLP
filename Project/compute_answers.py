#------------------------------------------------------------------------------
# File for using the trained models to get predictions
#------------------------------------------------------------------------------
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import pandas as pd
from functools import reduce
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import sys
from keras import backend as K
import os
import argparse

_EPSILON = 1e-7
def categorical_cross_entropy_loss(target, output):
    '''
        Loss function used in the model
    '''
    output /= tf.reduce_sum(output, -1, True)
    # manual computation of crossentropy
    epsilon = K.constant(_EPSILON, output.dtype.base_dtype)
    output = tf.clip_by_value(output, epsilon, 1. - epsilon)
    return - tf.reduce_sum(target * tf.math.log(output), -1)

def load_model(dir):
    '''
        - Load the trained model using keras.models.load_model
        - Load the tokenizer word index to give the words 
        the same index as in training.
        - Load MAX_SEQ_LEN used in training to pad testset 
        to the correct length.
        Outputs:
            Trained model
            Tokenizer word to index dictionary
            Max sequence length (int)
    '''
    print("\nLoading model...\n")
    model = keras.models.load_model(f'{dir}/model', custom_objects={'categorical_cross_entropy_loss':categorical_cross_entropy_loss})
    with open(f'{dir}/tokenizer.txt') as f:
        tokenizer_word_index = json.load(f)
    with open(f'{dir}/MAX_SEQ_LEN.txt') as f:
        MAX_SEQ_LEN = json.load(f)
    return model, tokenizer_word_index, MAX_SEQ_LEN


def get_test_data(path, tokenizer_word_index, MAX_SEQ_LEN):
    '''
        Loads testdata, makes it into a dataframe, lower the 
        text and strip text and tokenize the words using the 
        word_index dict from training and padds sequences.
        Input:
            Path to the testdata file
            tokenizer word index list from training
            MAX_SEQ_LEN (int) used in training
        Ouput:
            tokenized context (np.array)
            tokenized question (np.array)
            dataframe

    '''
    print(f'\nGet test data from {path}')
    # Import json file from path
    def load_json(dataset_path="./dataset/test.json"):
        '''Load testdata from json file'''    
        with open(dataset_path) as f:
            raw_json = json.load(f)

        return raw_json['data']

    def create_dataframe(data):
        '''Create dataframe of the given data'''
        contexts = []
        questions = []
        question_ids = []
        for i in range(len(data)):
            paragraphs = data[i]['paragraphs']
            for sub_para in paragraphs:
                for q_a in sub_para['qas']:
                    questions.append(q_a['question'])
                    question_ids.append(q_a['id'])
                    contexts.append(sub_para['context'])   
        df = pd.DataFrame({"questionID":question_ids, "context":contexts, "question": questions})
        return df

    def clean_text(dataframe):
        '''Make the text into lower and remove all leading and trailing whitespace'''
        def lower(text: str) -> str:
            return text.lower()
        def strip_text(text: str) -> str:
            return text.strip()  

        PREPROCESSING_PIPELINE = [
                            lower,
                            strip_text
                            ]

        def text_prepare(text: str) -> str:
            """
            Applies a list of pre-processing functions in sequence (reduce).
            """

            filter_methods = PREPROCESSING_PIPELINE
            if type(text) == list:
                new_row = [reduce(lambda txt, f: f(txt), filter_methods, x) for x in text]
            else:
                new_row = reduce(lambda txt, f: f(txt), filter_methods, text)
            return new_row
        for key in ['context', 'question']:
            dataframe[key] = dataframe[key].apply(lambda txt: text_prepare(txt))
        
        return dataframe

    def textToTensor(tokenizer, max_len, text):
        '''
            Converts text to tensors by converting the words into the correct indexes. 
            Then padds the tensors with 0 vlaues
        '''
        seq = tokenizer.texts_to_sequences(text)
        padded = pad_sequences(sequences=seq, maxlen=max_len, padding='post')
        return padded

    def tokenize(df, tokenizer_word_index, MAX_SEQ_LEN):
        '''Creates a tokenizer using the word_index dicitonary from training'''
        tokenizer = Tokenizer()
        tokenizer.word_index = tokenizer_word_index
        context = textToTensor(tokenizer, MAX_SEQ_LEN, df['context'])
        question = textToTensor(tokenizer, MAX_SEQ_LEN, df['question'])
        return context, question

    data = load_json(path)
    df = create_dataframe(data)
    df = clean_text(df)
    context, question = tokenize(df, tokenizer_word_index, MAX_SEQ_LEN)
    return context, question, df


def get_predicitons(model, context, questions):
    '''Use the model to predict on the testset'''
    print('\nGet predicitons..')
    predictions = model.predict([questions, context])
    print("\nGotten the predcitions!")
    return predictions


def make_answer_dict(start_preds, end_preds, df):
    '''Convert predicitons to a dicitonary containing question ID and answer text'''
    print('\nConvert predicitons to answer text..')
    def get_word_index(prediction):
        return [np.argmax(prediction[i]) for i in range(len(prediction))]

    def get_answer_text(start, end, index, df):
        '''Get answer text from context'''
        try:
            words = df['context'][index].split(' ')[start:end+1]
            answ = " ".join(words)
        except IndexError:
            print('When making answer, got index out of range')
            answ = ""
        return answ

    answer_dict = {}
    start_indxs = get_word_index(start_preds)
    end_indxs = get_word_index(end_preds)
    for i in range(len(start_preds)):
        question_id = df['questionID'][i]
        start_index = start_indxs[i]
        end_index = end_indxs[i]
        answr_text = get_answer_text(start_index, end_index, i, df)
        answer_dict[question_id] = answr_text
    return answer_dict

def write_predictions(answer_dict, path):
    '''Write answers to a prediciton file'''
    print(f'\nSaving answer to {path}')
    with open(path, 'w') as file:
     file.write(json.dumps(answer_dict))

def parse_args():
    '''Define parsing arguments'''
    parser = argparse.ArgumentParser('File for computing answers')
    parser.add_argument('test_data_file', metavar='testset.json', help='Input testset data JSON file.')
    parser.add_argument('--out-file', '-o', metavar='pred.txt',
                        help='Write accuracy metrics to file (default is stdout).')
    if len(sys.argv) == 0:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def main(test_path='./dataset/test.json', prediction_path='predict.txt'):
    ''' Main function '''
    # Path to where the newest model is stored
    model_folder='./models/model_30_12_2021_17_56_51'
    model, tokenizer_word_index, MAX_SEQ_LEN = load_model(model_folder)
    context, question, df = get_test_data(test_path, tokenizer_word_index, MAX_SEQ_LEN)
    pred_start, pred_end = get_predicitons(model, context, question)
    answer_dict = make_answer_dict(pred_start, pred_end, df)
    write_predictions(answer_dict, prediction_path)

if __name__ == "__main__":
    OPTS = parse_args()
    # Get path to test set
    test_path = OPTS.test_data_file
    if OPTS.out_file:
        # Get path to output prediciton file
        prediction_path = OPTS.out_file
    else:
        # Default value
        prediction_path = 'predictions.txt'
    main(test_path, prediction_path)