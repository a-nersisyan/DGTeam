# Got Systems?

## Table of Contents
* [General Info](#general-info)
* [GitHub Contents](#github-contents)
* [Technologies](#technologies)
* [Features](#features)
* [Classification Models](#classification-models)
* [Notebook Instructions](#notebook-instructions)
* [UI Instructions](#ui-instructions)
* [Inspiration](#inspiration)
* [Status](#status)

## General Info

***Problem:***

Today, there is heightened awareness around data privacy. Organizations struggle to comply with GDPR and CCPA regulations because data management systems were not created with privacy in mind. Failure to comply with said regulations may result in distrust in and damage to the brand. An ideal product would map the flow of data and the types of data between the systems a company uses. Because of how integrated software may be to some organizations, it is hard for them to narrow down from the metadata exactly what systems they are using. This issue happens at the lowest level of creating a data map. The current process to identify systems is time-consuming, expensive, and prone to human error.

***Solution:*** 

Our solution is a machine learning model that generates the names of the top three predicted systems, given some metadata. This model uses different similarity functions (Jaccard, Dice, Ratcliff-Obershelp, Jaro-Winkler and Cosine) to calculate the text similarity between input metadata and known system names. These text similarity scores are the features we use to train our classification models (KNeighbors, XGBoost, RandomForest, GaussianNB, DecisionTree, GradientBoosting). Our model gathers the predicted systems for each of these classifiers and, through voting methods, returns the top three predictions. 

[(Back to top)](#table-of-contents)

## Github Contents

***lib:***
This folder contains the func.py file. This file contains the functions needed to create a class instance for each core system in the notebook. 

***models:***
This folder contains pre-trained models for each core system, which were saved as joblib files using pickle. 

***DG_UI:***
This file contains the code needed to produce the UI. 

[(Back to top)](#table-of-contents)

## Technologies

* Language: Python 3
* Libraries: pandas, sklearn, nltk, strsimpy, matplotlib, numpy, textdistance, re

[(Back to top)](#table-of-contents)

## Features

Similarity scores generated by similarity functions were used as the features to feed into the model. In our case, we used five similarity functions.

* Jaccard
* Dice
* Ratcliff-Obershelp
* Jaro-Winkler
* Cosine 

[(Back to top)](#table-of-contents)

## Classification Models

Classifiers were used to create predictions. In our case, we used six classification models.

* K-Nearest Neighbors
* XGBoost
* Random Forest
* Gaussian Naive Bayes
* DecisionTree
* Gradient Boosting
* ~~Logistic Regression~~ Outputted too many systems.
* ~~Support Vector Machine~~ Slow runtime.

[(Back to top)](#table-of-contents)

## Notebook Instructions

***If using our core systems:***
1. Make a directory folder called DGTeam to store the code from github.
2. Clone the github repository to your local repository.
3. Upload the data, lib, and matrices folders to Deepnote.
4. Create a new notebook for each core system within the same directory.
5. Run the following two lines within each notebook:
    ```
    import pandas as pd
    from lib import funcs
    ```
6. Create a class instance for each core system by running `[core system] = funcs.Person('[core system]_data')` within each notebook.
7. Run `[core system].table = pd.read_csv([path to core system's matrix in matrices folder])` within each notebook to initialize the table variable.
8. Run `[core system].model_maker()` within each notebook to initialize the classification models. 
9. Run `[core system].PR()` within each notebook to display the precision-recall curve for the classification models. 
10. Run the following lines within each notebook to display the top predictions for a given query:
    ```
    top_predicted_systems = [core system].official_model([query])
    top_predicted_systems
    ```
***If using new core systems:***
1. Make a directory folder called DGTeam to store the code from github.
2. Clone the github repository to your local repository.
3. Upload the folders to a Jupyter Notebook or Deepnote directory.
4. Add the csv for each new core system to the data folder. Make sure they have the same three columns as our core system datasets. This means that if your dataset originally comes with just metadata, you have to manually label the dataset.
5. Create a new notebook for each core system within the same directory.
6. Within funcs.py under "Import datasets", assign a new global variable for each new core system using the format:
    ```
    [core system]_data = pd.read_csv([path to new dataset in data folder]).drop_duplicates(subset=['label']).reset_index(drop=True)
    ```
7. Within funcs.py under "Making a list of all known systems", assign a global variable in the format: 
    ```
    [core system]_systems = [core system]_data[['system']]
    ```
    and append this to the labeled_systems column.
8. Within funcs.py under "Cleaning", clean the label column for each new core system using the format:
    ```
    [core system]_data['label'] = [core system]_data['label'].apply(clean).str.lower().apply(acronyms)
    ```
9. In the notebooks, run the following two lines:
    ```
    import pandas as pd
    from lib import funcs
    ```
10. Create a new code block within the thresholds function using the format:
    ```
    elif self.name == '[core system]_data':
            if sim_func == Person.jaccard:
                return 0.9
            elif sim_func == Person.dice:
                return 0.9
            elif sim_func == Person.rat:
                return 0.9
            elif sim_func == Person.jaro:
                return 0.9
            elif sim_func == Person.cosine:
                return 1
    ```
11. Create a class instance for each new core system by running `[core system] = funcs.Person('[core system]_data')` within each notebook.
12. Run `[core system].table = [core system].[core system]_data` within each notebook to initialize the table variable.
13. Run `[core system].system_binary_score_top3()` within each notebook to create the comprehensive dataframe. 
14. Within each notebook, save the comprehensive dataframe as a zip by running:
    ```
    compression_opts = dict(method='zip', archive_name='[core system]_matrix.csv')
    [core system].table.to_csv('[core system]_matrix.zip', index=False, compression=compression_opts)
    ```
15. Unzip the file and move the resulting csv into the matrices folder.
16. Run `[core system].table = pd.read_csv([path to core system's matrix in matrices folder])` within each notebook to reassign the table variable.
17. Run `[core system].model_maker()` within each notebook to initialize the classification models. 
18. Run `[core system].PR()` within each notebook to display the precision-recall curve for the classification models. 
19. Run the following lines within each notebook to display the top predictions for a given query:
    ```
    top_predicted_systems = [core system].official_model([query])
    top_predicted_systems
    ```
    
[(Back to top)](#table-of-contents)

## UI Instructions

![Screen Shot 2020-12-11 at 2 37 58 PM](https://user-images.githubusercontent.com/56287414/101961412-89ffcd80-3bbe-11eb-9829-20bff26bae48.png)


In order to increase the speed time of the interface, we saved the trained models as separate joblib files using pickle. We then load those saved files whenever a search is entered into the query and output the results using a drop-down menu included in Streamlit’s library.
 
***If using our core systems:*** 

1. Make a directory folder called DGTeam to store the code from github.
2. From your local terminal run the following command to make sure you have the software needed to load the UI: `pip intsall streamlit`, `pip intsall textdistance`, `pip intsall strsimpy`, `pip intsall xgboost==1.2.1`, `brew intsall libomp`, `pip intsall ipynb`,  `pip intsall nbimporter`, `pip intsall nltk`.
3. Clone the github repository to your local repository.
4. Navigate into the directory folder you have cloned the github repository into.
5. From here, write the following code in your terminal and a local server will show up on a new tab in your default browser:
    `streamlit run DG_UI.py`.
6. In the first drop down menu you can click into it and see all the core systems currently loaded into the UI. 
7. Input a query under the search bar below “Enter the query”.
8. If the trained model for the selected core system predicts a system within the query, a drop-down list of the top three predictions will appear. If there is no match the UI will simply state “This system name could not be found in this database.”

***If using new core systems:***

1. From your terminal run `pip install pickle` to install the serialization software for the model.
2. Open the `DG_UI.py` file with your local text or code editor of choice. 
3. Add the new core system name `core_system_name` to the system_options array.
4. Follow all the instructions to add a new core system to a notebook.
5. Under the Trained Models section, add the following code as it applies to the new core system:
```
if option1 == "core_system_name":
    core_name = Person("core_system_name_data")
    core_name.matrix = pd.read_csv('matrices/core_name.matrix.csv') To get the matrix.csv run the commented out with core_name.ensemble_matrix() ensemble_matrix
    core_name.table = core_name.okta_matrix
    core_name.model_maker()
    core_name.PR()
    dump(core_name, 'models/core_name_model.joblib')
```
6. Under the Loaded Models section, load the new core system model with the following line of code: 
`core_name_model = load('models/core_name_model.joblib')`
7. Under the options to set the Global Variable, write an if statement to set the global variable to the serialized model using: 
```
if option1 == "core_system_name":
     predict_database = core_name_model
```
8. From your terminal, input `streamlit run DG_UI.py` into the command line and run your updated UI.

[(Back to top)](#table-of-contents)

## Inspiration

Inspiration for this project comes from DataGrail.

[(Back to top)](#table-of-contents)

## Status

The status of this project is finished. However, future work would include using web scraping to create a more comprehensive systems database.

[(Back to top)](#table-of-contents)
