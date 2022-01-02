# NLP Project

The goal of this project is to create a NLP system that, given a paragraph and a question regard-
ing it, provides a single answer, which is obtained selecting a span of text from the paragraph.

## How to run
- cd into the project folder
- run `pip install -r requirements.txt` to install requirements
- run `python3 compute_answers.py path_to_test_dataset` to compute predicitons. Predicitons are saved to `predictions.txt`.
    - optionally run `python3 compute_answers.py path_to_test_dataset -o path_to_output_prediction_file`
- run `python3 evaluate.py path_to_ground_truth path_to_prediction_file`
