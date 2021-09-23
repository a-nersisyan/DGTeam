#### Import libraries ####
# Call these lines to import the necessary libraries.
import pandas as pd
import numpy as np
import textdistance
import sklearn
import re
import math
import matplotlib.pyplot as plt
import nltk
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import jaccard_score
from strsimpy.jaro_winkler import JaroWinkler
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

#### Define the Person class ####
# This is a necessary step if you intend to create a class instance for each core system in a separate notebook. 
# It prevents your computer from slowing down and makes it easier to apply all the main functions to more core systems.
class Person:
    
    #### Miscellaneous Variables ####
    # Global variables that can be called within class functions. It prevents future errors from occurring, since a lot of functions access similar features and classifiers. 
    # In case you need to select certain features or classifiers, you can simply change the global variable.
    classifiers = [KNeighborsClassifier, xgb, RandomForestClassifier, GaussianNB, DecisionTreeClassifier, GradientBoostingClassifier]            

    #### Import datasets ####
    # Import the initial datasets for each core system using pandas. The variable name should follow this format: [core system name]_data.
    google_data = pd.read_csv('/home/jovyan/work/data/google.csv').drop_duplicates(subset=['label']).reset_index(drop=True)
    sf_data = pd.read_csv('/home/jovyan/work/data/salesforce.csv').drop_duplicates(subset=['label']).reset_index(drop=True)
    hubspot_data = pd.read_csv('/home/jovyan/work/data/hubspot.csv').drop_duplicates(subset=['label']).reset_index(drop=True)
    okta_data = pd.read_csv('/home/jovyan/work/data/okta.csv').drop_duplicates(subset=['label']).reset_index(drop=True)
    marketo_data = pd.read_csv('/home/jovyan/work/data/marketo.csv').drop_duplicates(subset=['label']).reset_index(drop=True)
    # Import new dataset for each new core system here

    #### Making a list of all known systems ####
    # Since the intial datasets were manually labeled, you want to update your original systems dataset with any new ones that were found.
    google_systems = google_data[['system']]
    sf_systems = sf_data[['system']]
    hubspot_systems = hubspot_data[['system']]
    okta_systems = okta_data[['system']]
    marketo_systems = marketo_data[['system']]
    systems_yml = pd.read_csv('/home/jovyan/work/data/services_new.csv').drop_duplicates()
    # Define new [core system]_systems variables here
    
    labeled_systems = pd.DataFrame(columns=['system'])
    labeled_systems = labeled_systems.append([google_systems, sf_systems, hubspot_systems, okta_systems, marketo_systems]) # Append new [core system]_systems variables to this list
    labeled_systems = labeled_systems.dropna().drop_duplicates()
    labeled_systems = labeled_systems.rename(columns={"system": "display_name"})
    labeled_systems.index = pd.RangeIndex(len(labeled_systems.index))
    systems_series = pd.merge(systems_yml, labeled_systems, how='outer', on='display_name')
    systems = systems_series['display_name'].drop_duplicates().tolist()
   
    #### init method or constructor ####  
    def __init__(self, name):   
        self.name = name


    #### Cleaning ####
    # Apply clean to the label column of Okta, Google, Marketo, and Hubspot to remove punctuation marks and camel case.
    # Apply clean_sf to the label column of Salesforce to remove punctuation marks and camel case.
    # Apply acronyms to the label column to replace acronyms with the full name of the system.
    def clean(label):
        label = label.replace('_', ' ')
        m = re.search(r'\b(iOS|DiscoverOrg|1Password|mParticle|EnvirOmatic|InMoment|McAfee|CLink|FedEx|eSpark|TINYpulse|PlayVox|MicroStrategy|FunRetro|AllTrails|ScaleFT|PayPal|LinkedIn|DataGrail|iCivics|macOS|SEMrush|NoRedInk|SendGrid|DocuSign|BlazeMeter|RealtimeBoard|FireTube|WalkMe|SendinBlue|SalesLoft|SurveyMonkey|BrandVia|HubSpot|ActiveCampaign|AnswerHub|SharpSpring|FullStory)(\b|[A-Z])', label)
        if m:
            g = m.group(1) if len(m.groups()) > 1 else m.group(0)
            label = label.replace(g, g.lower())
        label = re.sub('([A-Z][a-z]+)', r'\1', re.sub('([A-Z]+)', r'\1', label))
        rep = ["&",".com",".org",".io", "-", "@", ".", "(", ")"] #removed underscore from list
        for replace in rep:
            label = label.replace(replace, " ")
        return label.lower()

    def clean_sf(label):
        label = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r'\1', label)).strip()
        label = label.replace(r"  +", " ")
        rep = ["&",".com",".org",".io", "-", "@", ".", "(", ")", "_"]
        for replace in rep:
            label = label.replace(replace, " ")
        return label

    def acronyms(label):
        m = re.search(r'\b(hs)\b', label)
        if m:
            label = label.replace(m.group(0),'hubspot')

        m = re.search(r'\b(sfdc)\b', label)
        if m:
            label = label.replace(m.group(0),'salesforce')

        m = re.search(r'\b(aws)\b', label)
        if m:
            label = label.replace(m.group(0),'amazon web services')

        m = re.search(r'\b(cbt)\b', label)
        if m:
            label = label.replace(m.group(0),'cloud bigtable')

        m = re.search(r'\b(amz)\b', label)
        if m:
            label = label.replace(m.group(0),'amazon')

        m = re.search(r'\b(mk)\b', label)
        if m:
            label = label.replace(m.group(0),'mk getler')

        m = re.search(r'\b(es)\b', label)
        if m:
            label = label.replace(m.group(0),'everstring')

        m = re.search(r'\b(clink)\b', label)
        if m:
            label = label.replace(m.group(0),'centurylink')

        m = re.search(r'\b(meraki)\b', label)
        if (m) and not (re.search(r'\b(cisco)\b', label)):
            label = label.replace(m.group(0),'cisco meraki')
        
        return label

    google_data['label'] = google_data['label'].apply(clean).str.lower().apply(acronyms)
    hubspot_data['label'] = hubspot_data['label'].apply(clean).str.lower().apply(acronyms)
    sf_data['label'] = sf_data['label'].apply(clean_sf).str.lower().apply(acronyms)
    okta_data['label'] = okta_data['label'].apply(clean).str.lower().apply(acronyms)
    marketo_data['label'] = marketo_data['label'].apply(clean).str.lower().apply(acronyms)
    # Clean column for new core systems here
    
    #### N-grams ####
    # ngram_maker produces a list of all grams up to a certain n for each label. 
    # This is needed for the rat and jaro functions, since they require different tokenization than the other similarity functions.
    def ngram_maker(label, n):
        label_split = label.split()

        label_tuples = list(ngrams(label_split, n))
        label_list = [' '.join(w) for w in label_tuples]
        
        return label_list
        
    #### Similarity Functions ####
    # Function that calculates the jaccard similarity for dataset labels
    def jaccard(label, system):
        label, system = set(label.split()), set(system.lower().split())
        intersection = (label).intersection(system)
        union = (label).union(system)
        score = float(len(intersection)) / len(union)
        return score

    # Function that calculates the dice similarity for dataset labels
    def dice(label, system):
        jaccard_score = Person.jaccard(label, system)
        return 2*jaccard_score/(1+jaccard_score)

    # Function that calculates the ratcliff-obershelp similarity for dataset labels
    def rat(label, system):

        label_len = len(label.split())
        system_len = len(system.split())
        scores = []
        if system_len >= label_len:
            score = textdistance.ratcliff_obershelp(label, system.lower())
            scores.append(score)
        else:
            n = system_len
            label_list = Person.ngram_maker(label, n)
            for label_gram in label_list:
                score = textdistance.ratcliff_obershelp(label_gram, system.lower())
                scores.append(score)
        return max(scores)

    # Function that calculates the jaro winklar similarity for dataset labels
    def jaro(label, system):
        jarowinkler = JaroWinkler()
        label_len = len(label.split())
        system_len = len(system.split())
        scores = []
        if system_len >= label_len:
            score = jarowinkler.similarity(label, system)
            scores.append(score)
        else:
            n = system_len
            label_list = Person.ngram_maker(label, n)
            for label_gram in label_list:
                score = jarowinkler.similarity(label_gram, system)
                scores.append(score)
        return max(scores)

    # Function that calculates the cosine similarity for dataset labels
    def cosine(label, system):
        count_vectorizer = CountVectorizer()
        sparse_matrix = count_vectorizer.fit_transform([label,system])
        string_matrix = sparse_matrix.todense()
        df = pd.DataFrame(string_matrix, 
                  columns=count_vectorizer.get_feature_names(), 
                  index=['label', 'system']) 
        score = cosine_similarity(df,df)
        return score[0][1]

    #### Thresholds for each similarity function ####
    # Threshold sets the thresholds for each similarity function and if the similarity score is above the threshold, 
    # the system_binary_score_top3 function will predict the label as having a system and return the name of that system.
    def threshold(self, sim_func):
        if self.name == 'marketo_data':
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
        
        elif self.name == 'google_data':
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

        elif self.name == 'sf_data':
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

        elif self.name == 'hubspot_data':
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

        elif self.name == 'okta_data':
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
       
        # Add  code block for new core systems here
    
    #### Adding Features ####
    # Call system_binary_score_top3 to add columns to the dataframe. The columns indicating the top score of each similarity function form the design matrix.
    def system_binary_score_top3(self):
        sim_funcs = [Person.rat, Person.cosine, Person.jaro, Person.jaccard, Person.dice]
        for sim_func in sim_funcs:
            system_col_name = '{} system'.format(sim_func.__name__)
            binary_col_name = '{} binary'.format(sim_func.__name__)
            scores_col_name = '{} score'.format(sim_func.__name__)

            labels = self.table['label'].drop_duplicates()
            pairs1 = {}
            pairs2 = {}
            pairs3 = {}
            for label in labels:
                pairs_list = []
                for system in Person.systems:
                    score = sim_func(label, system)
                    pairs_list.append((score, system))
                pairs_list.sort(reverse=True)
                pairs1[label] = pairs_list[0]
                pairs2[label] = pairs_list[1]
                pairs3[label] = pairs_list[2]
            all_pairs = [pairs1, pairs2, pairs3]

            self.table[system_col_name] = all_pairs[0].values()
            self.table[scores_col_name] = np.nan
            for index, pair in self.table[system_col_name].iteritems():
                if pair[0] >= Person.threshold(self, sim_func):
                    self.table[system_col_name].loc[index] = pair[1]
                    self.table[scores_col_name].loc[index] = pair[0]
                else:
                    self.table[system_col_name].loc[index] = np.nan
                    self.table[scores_col_name].loc[index] = pair[0]

            self.table[binary_col_name] = np.nan
            for index, system in self.table[system_col_name].iteritems():
                self.table[binary_col_name].loc[index] = (type(system) != float)*1
            
            for i in range(3):
                self.table[system_col_name+' {}'.format(i+1)] = all_pairs[i].values()
                for index, pair in self.table[system_col_name+' {}'.format(i+1)].iteritems():
                    self.table[system_col_name+' {}'.format(i+1)].loc[index] = pair[1]

    #### Classifier Building ####
    # Call model_maker to build classification models based on selected top score features.
    def model_maker(self):
        self.X = self.table[['rat score', 'cosine score', 'jaro score', 'jaccard score', 'dice score']]
        self.y = self.table['binary']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X.values, self.y.values, test_size=0.2, random_state=42)

        for classifier in Person.classifiers:
            if classifier == KNeighborsClassifier:
                self.modelKNeighbors = classifier(n_neighbors=3)
                self.modelKNeighbors.fit(self.X_train, self.y_train)
            elif classifier == xgb:
                self.modelxgb = classifier.XGBClassifier()
                self.modelxgb.fit(self.X_train, self.y_train)
            elif classifier == RandomForestClassifier:
                self.modelRandomForest = classifier(n_estimators=20)
                self.modelRandomForest.fit(self.X_train, self.y_train)
            elif classifier == GaussianNB:
                self.modelGaussianNB = classifier()
                self.modelGaussianNB.fit(self.X_train, self.y_train)
            elif classifier == DecisionTreeClassifier:
                self.modelDecisionTree = classifier()
                self.modelDecisionTree.fit(self.X_train, self.y_train)
            elif classifier == GradientBoostingClassifier: 
                self.modelGradientBoosting = classifier(random_state=0)
                self.modelGradientBoosting.fit(self.X_train, self.y_train)
            else:
                self.model = classifier()
                self.model.fit(self.X_train, self.y_train)

    #### PR Curves ####
    # Call PR to draw precision-recall curves for the classifiers to evaluate their performance.
    def PR(self):
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision v. Recall')

        models = [self.modelKNeighbors,self.modelxgb, self.modelRandomForest, 
        self.modelGaussianNB, self.modelDecisionTree, self.modelGradientBoosting]

        colors = ['red', 'blue', 'green', 'orange', 'cornflowerblue','plum']
        
        model_colors_pairs = zip(models, colors)
        for model, color in model_colors_pairs:
            y_pred = model.predict_proba(self.X_test)
            pos_probs = y_pred[:, 1]
            precision, recall, _ = precision_recall_curve(self.y_test, pos_probs)
            plt.plot(recall, precision, color, label="{}".format(type(model).__name__))
        
        plt.legend()
        plt.show()


    #### Prediction Generator ####
    # Call official_model(query) to generate the predicted system names for given query.
    def official_model(self, query):
        query = Person.clean(query)
        query = Person.acronyms(query)

        X_list = []
        for system in Person.systems:
            system = system.lower()
            scores_list = [Person.rat(query,system), Person.cosine(query, system), 
            Person.jaro(query,system), Person.jaccard(query, system), Person.dice(query, system)]
            X_list.append(scores_list)

        design_matrix = np.array(X_list)

        predictionKNeighbors = self.modelKNeighbors.predict(design_matrix)
        predictionXGB = self.modelxgb.predict(design_matrix)
        predictionRandomForest = self.modelRandomForest.predict(design_matrix)
        predictionGaussianNB = self.modelGaussianNB.predict(design_matrix)
        predictionDecisionTree = self.modelDecisionTree.predict(design_matrix)
        predictionGradientBoosting = self.modelGradientBoosting.predict(design_matrix)


        predicted_systemsKNeighbors = []
        predicted_systemsXGB = []
        predicted_systemsRandomForest = []
        predicted_systemsGaussianNB = []
        predicted_systemsDecisionTree = []
        predicted_systemsGradientBoosting = []
        
        for i in range(len(predictionKNeighbors)):
            if predictionKNeighbors[i] == 1:
                predicted_systemsKNeighbors.append(self.systems[i])
            if predictionXGB[i] == 1:
                predicted_systemsXGB.append(self.systems[i])
            if predictionRandomForest[i] == 1:
                predicted_systemsRandomForest.append(self.systems[i])                
            if predictionGaussianNB[i] == 1:
                predicted_systemsGaussianNB.append(self.systems[i])
            if predictionDecisionTree[i] == 1:
                predicted_systemsDecisionTree.append(self.systems[i])               
            if predictionGradientBoosting[i] == 1:
                predicted_systemsGradientBoosting.append(self.systems[i])

        models = [self.modelKNeighbors, self.modelxgb, 
                self.modelRandomForest, self.modelGaussianNB,
                self.modelDecisionTree, self.modelGradientBoosting]

        predicted_systems = [predicted_systemsKNeighbors,                                           
                            predicted_systemsXGB,
                            predicted_systemsRandomForest,
                            predicted_systemsGaussianNB,
                            predicted_systemsDecisionTree,
                            predicted_systemsGradientBoosting]       

        combined_predictions = []
        for list_of_systems in predicted_systems:
            combined_predictions += list_of_systems
        
        system_counter = {}
        if len(combined_predictions) != 0:
            for system in combined_predictions:
                if system in system_counter:
                    system_counter[system] += 1
                else:
                    system_counter[system] = 1
            popular_words = sorted(system_counter, key = system_counter.get, reverse = True)
            rows = []
            if len(popular_words) >= 3:
                top_3 = popular_words[:3]
                for i in range(3):
                    rows += Person.systems_series.loc[Person.systems_series['display_name'] == popular_words[i]].values.tolist()
                return rows
            else:
                top = popular_words
                for i in range(len(top)):
                    rows += Person.systems_series.loc[Person.systems_series['display_name'] == popular_words[i]].values.tolist()
                return rows

        
