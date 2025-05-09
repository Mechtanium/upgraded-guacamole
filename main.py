import functions_framework
import warnings
warnings.filterwarnings('ignore')
from flask import Flask, jsonify, make_response
import numpy as np
# from sentence_transformers import SentenceTransformer
import random
import itertools
import tensorflow as tf
import os
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import requests

REFRESH_SIZE = 50
PATIENCE = 10
DELTA = 0.01

TRAINING = 1
TERMINATED = 2
FINISHED = 3

loss_history = []
test_loss_history = []

server_error = lambda e: jsonify({
    "error": "Server error",
    "message": e
})
smoothener = lambda alpha, value, arr: (alpha * arr[-1]) + ((1 - alpha) * value) if (len(arr) > 0) else value


class MyCallback(tf.keras.callbacks.Callback):
    def __init__(self, question_id, **kwargs):
        super().__init__(**kwargs)
        self.question_id = question_id

    def on_epoch_end(self, epoch, logs=None):
        global loss_history, test_loss_history
        url = os.environ.get('state_update')
        if epoch % REFRESH_SIZE == 0:
            result = requests.post(
                url if url else "www.google.com",
                params={
                    'question_id': self.question_id,
                    'train_loss':loss_history[-REFRESH_SIZE:],
                    'test_loss':test_loss_history[-REFRESH_SIZE:]
                }
            )

            status_url = os.environ.get('status')
            if status_url:

                status = requests.get(
                    status_url if status_url else "www.google.com",
                    params={'question_id': self.question_id}
                ).json()["train_status"]

                print(status)
                if status == TERMINATED:
                    self.model.stop_training = True
        loss_history.append(smoothener(0.9, logs['loss'], loss_history))
        test_loss_history.append(smoothener(0.9, logs['val_loss'], test_loss_history))

    def on_train_end(self, logs=None):
        end_train_url = os.environ.get('on_train_end')
        requests.post(
            end_train_url if end_train_url else "www.google.com",
            params={ 'question_id': self.question_id }
        )
        saved = save_model_to_gcs(self.model, self.question_id)

        if not saved:
            raise RuntimeError(server_error("Unable to save model to GCS"), 511)


def train_test_split(responses):
    dimen = len(responses)
    quarter = dimen // 4
    train = responses[:dimen - quarter - 1]
    test = responses[dimen - quarter:]
    return train, test


def encode_list(model, responses):
    return np.array([model.encode(item["response"]) for item in responses])


def merge_ideal(model, question, encode_list):
    ideal_embed = np.array(model.encode(question["ideal_response"]))
    return np.array([np.concatenate([item, ideal_embed]) for item in encode_list], dtype=np.float64)


def cascading_count(top_array):
    count = 0
    for item in top_array:
        count += len(item)

    return count


def random_sampling(merged_Xy):
    dataset = []

    # check that the total count of items in items in list
    while cascading_count(merged_Xy) > 0:
        # prevent error when remaining items are less than 4
        if len(merged_Xy) <= 3:
            break
        # get 4 random ids from within the range of the available items in list
        # the random.sample function inherently samples unique items
        ids = random.sample(range(len(merged_Xy)), 4)

        # for each id in ids, select the first item in the list item at position id of list
        for idx in ids:
            dataset.append(merged_Xy[idx].pop(0))

        # sort ids in reverrse so removing items does not change position of other items
        ids.sort(reverse=True)

        # if list item in list is empty, remove it so the list shrinks in next iteration
        for idx in ids:
            if len(merged_Xy[idx]) == 0:
                merged_Xy.pop(idx)

    # shuffle and add what's left to the dataset list
    flattened_list = list(itertools.chain(*merged_Xy))
    random.shuffle(flattened_list)
    dataset += flattened_list

    return dataset


def prepare_Xy(dataset):
    X = []
    real_y = []
    for row in dataset:
        x, y = row
        X.append(x)
        real_y.append(y)

    X = np.array(X)
    Y = np.array(real_y)

    return X, Y


def train_model(X, y, question):
    # Ensure that TensorFlow is using the GPU
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("Using GPU")
    else:
        print("No GPU found, using CPU")

    grade_rationalizer_model = Sequential([
        Dense(2, activation='sigmoid', input_shape=(X.shape[1],)),
        Dense(1, activation='relu')
    ])

    url = os.environ.get('sk_base_artifact')
    file_path = 'base_model.weights.h5'

    if (url):
        tf.io.gfile.copy(url, file_path, overwrite=True)
        grade_rationalizer_model.load_weights(file_path)

    grade_rationalizer_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='mean_squared_error')

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=DELTA,
        patience=PATIENCE,
        verbose=1,
        mode='auto',
        baseline=None,
        restore_best_weights=True,
        start_from_epoch=1
    )

    history = grade_rationalizer_model.fit(X, y, epochs=1000, batch_size=3,
                                                validation_split=0.3, callbacks=[
                                                MyCallback(question['_id']),
                                                early_stopping ]
                                            )

    return history.history, grade_rationalizer_model


def save_model_to_gcs(model, question_id):
    try:
        """Save TensorFlow model to Google Cloud Storage."""
        # Save the model to a temporary directory
        file_path = f'{question_id}.weights.h5'
        gcs_path = f"gs://model-leon-279400-store/models/v1/{file_path}"

        model.save_weights(file_path)
        tf.io.gfile.copy(file_path, gcs_path, overwrite=True)

    except Exception as e:
        print(f"Error saving model to GCS: {e}")
        return False

    return True


def train_job(graded_data, question_data):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')

    graded_train, graded_test = train_test_split(graded_data)

    encoded_graded_train = encode_list(model, graded_train)

    encoded_graded_test = encode_list(model, graded_test)

    train_X = merge_ideal(model, question_data, encoded_graded_train)

    test_X = merge_ideal(model, question_data, encoded_graded_test)

    train_y = np.array([item["grade"] for item in graded_train], dtype=np.float64).reshape(-1, 1)

    merged_Xy_train = [[(X, y)] * 4 for X, y in zip(train_X, train_y)]

    train_data = random_sampling(merged_Xy_train)

    X, Y = prepare_Xy(train_data)

    history, model = train_model(X, Y, question_data)

    pred_graded_train = model.predict(train_X)

    for idx, item in enumerate(graded_train):
        item["pred_grade"] = float(pred_graded_train[idx][0])

    pred_graded_test = model.predict(test_X)

    for idx, item in enumerate(graded_test):
        item["pred_grade"] = float(pred_graded_test[idx][0])

    return {
        "status": True,
        "history": history,
        "graded": {
            "train": graded_train,
            "test": graded_test
        }
    }


def grade(ungraded_data, question_data):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L12-v2')

    encoded_ungraded = encode_list(model, ungraded_data)

    ungraded = merge_ideal(model, question_data, encoded_ungraded)

    grade_rationalizer_model = Sequential([
        Dense(2, activation='sigmoid', input_shape=(ungraded.shape[1],)),
        Dense(1, activation='relu')
    ])

    URI = os.environ.get('sk_URI')

    file_path = f'{question_data["_id"]}.weights.h5'
    gcs_path = f"{URI}{file_path}"

    tf.io.gfile.copy(gcs_path, file_path, overwrite=True)
    grade_rationalizer_model.load_weights(file_path)

    pred_ungraded = grade_rationalizer_model.predict(ungraded)

    for idx, item in enumerate(ungraded_data):
        item["pred_grade"] = float(pred_ungraded[idx][0])

    return {
        "status": True,
        "ungraded": ungraded_data
    }


@functions_framework.http
def train_grader(request):
    try:
        request_json = request.get_json()
        request_args = request.args

        if request_json and 'data' in request_json \
        and 'graded_responses' in request_json['data'] \
        and 'question' in request_json['data']:
            graded_data = request_json['data']['graded_responses']
            question_data = request_json['data']['question']

            return make_response(jsonify(train_job(graded_data, question_data)), 200)

        elif request_json and 'data' in request_json \
        and 'responses' in request_json['data'] \
        and 'question' in request_json['data']:
            ungraded_data = request_json['data']['responses']
            question_data = request_json['data']['question']

            return make_response(jsonify(grade(ungraded_data, question_data)), 200)

        else:
            graded_data = None
            ungraded_data = None
            question_data = None

        if not isinstance(graded_data, list) or not isinstance(ungraded_data, list) or not question_data:
            return make_response(jsonify({
                "error": "Bad request",
                "message": f'''Invalid input arguments/structure:
                request: {request},
                json: {request_json},
                graded_data : {graded_data},
                ungraded_data : {ungraded_data},
                question_data: {question_data}'''
            }), 400)


    except Exception as e:
        return make_response(server_error(str(e)), 500)