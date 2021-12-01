# Disaster Response Pipeline Project

### Table of Contents
1. [Brief Summary](#summary)
2. [Project Description](#proj_desc)
3. [Files Descriptions](#file_desc)
4. [Running Instructions](#instructions)

### Brief Summary<a name="summary"></a>
This project is about training Machine Learning Model based on Random Forest Classifier to classify words in disaster messages into 36 predetermined categories

This to help direct a disaster message to the appropriate channel for further support.

### Project Description<a name="proj_desc"></a>
Project has three parts:

1. **ETL Pipeline:** `process_data.py` This part contains the ETL pipeline script that `messages` and `categories` datasets, clean and merge them and store the data into SQLite database file 

2. **ML Pipeline:** `train_classifier.py` This part contains ML pipeline that load the data from SQLite databased, prepare the data, instantize and train ML model using GridSearchCV with different parameters and export the model to pickle file

3. **Flask Web App:** `run.py` This part will display three visualizaions from loaded messages dataset and has an input to let users enter a message then categorize it based on the saved model

### Files Descriptions<a name="file_desc"></a>f
	- README.md: This file
	- workspace
		- \app
			- run.py: python file to run the app
			- \templates
				- master.html: main webpage
				- go.html: result webpage
		- \data
			- disaster_categories.csv: categories dataset
			- disaster_messages.csv: messages dataset
			- DisasterResponse.db: generated disaster response database
			- process_data.py: ETL pipeline python code
		- \models
			- train_classifier.py: ML pipeline python code


### Running Instructions <a name="instructions"></a>
Instructions to setup the database and ML model then run the web app:

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

