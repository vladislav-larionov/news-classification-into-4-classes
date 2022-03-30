
# Preparation of the environment

All the commands in the following sections should be executed only after appropriate initialization of the application environment. The instructions below are related mostly to Linux. On Windows modify the commands accordingly. The name of the interpreter can be `python` or `python3` depending on the system.

Create a new virtual environment if it doesn't exist:
```
python3 -m venv venv
```

Activate the virtual environment:
```
source venv/bin/activate
```
or, on Windows:
```
venv\Scripts\activate
```

Install requirements (first time or after modification of `requirements.txt`):
```
pip3 install -r requirements.txt
```

launch:
```
python3 topic_classifier.py --test_data_source mention_lemmed_title_text_no_stopwords_w --learning_data_source mention_lemmed_title_text_no_stopwords_w --use_whole_text
```