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


def joint_loss(y_true, y_pred):
    '''
    Define loss function used in the trained model
    '''
    loss_start = tf.keras.losses.CategoricalCrossentropy()(y_true[0], y_pred[0])
    loss_end = tf.keras.losses.CategoricalCrossentropy()(y_true[1], y_pred[1])
    #tf.print("Pred start", y_pred[0])
    return loss_start + loss_end

# Load model
def load_model(dir='./models/model_21_12_2021_15_41_21'):
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
    print("Loading model...")
    model = keras.models.load_model(f'{dir}/model', custom_objects={'joint_loss': joint_loss})
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
    print(f'Get test data from {path}')
    # Import json file from path
    def load_json(dataset_path="training_set.json"):
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
    print('Get predicitons..')
    predictions = model.predict([questions, context])
    return predictions


def make_answer_dict(start_preds, end_preds, df):
    '''Convert predicitons to a dicitonary containing question ID and answer text'''
    print('Convert predicitons to answer text..')
    def get_word_index(prediction):
        return [np.argmax(prediction[i]) for i in range(len(prediction))]

    def get_answer_text(start, end, index, df):
        '''Get answer text from context'''
        words = df['context'][index].split(' ')[start:end]
        answ = " ".join(words)
        # NB!!!: fore some reason the end is projected to be before the start so the answers are empty strings. 
        # Just doing this for now.
        if answ == "":
            print("\n\nSOMETHING STRANGE IS GOING ON AND END PRED IS BEFORE START PRED\n\n")
            answ = df['context'][index].split(' ')[start]
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
    print(f'Saving answer to {path}')
    with open(path, 'w') as file:
     file.write(json.dumps(answer_dict))

def main(test_path, prediction_path):
    model, tokenizer_word_index, MAX_SEQ_LEN = load_model()
    context, question, df = get_test_data(test_path, tokenizer_word_index, MAX_SEQ_LEN)
    pred_start, pred_end = get_predicitons(model, context, question)
    answer_dict = make_answer_dict(pred_start, pred_end, df)
    write_predictions(answer_dict, prediction_path)

if __name__ == "__main__":
    # Default values
    test_path = './SQUAD MATERIAL/training_set_copy.json'
    prediction_path = 'predictions.txt'
    if len(sys.argv) > 1:
        test_path = sys.argv[1]
        if len(sys.argv) > 2:
            prediction_path = sys.argv[2]
    main(test_path, prediction_path)