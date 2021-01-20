# Skill Equivalency Learning

This repo contains the datasets and code for the 2021 LAK paper **Learning Skill Equivalencies Across Platform Taxonomies**.


## Files and folders

[example.sh](https://github.com/CAHLR/skill-equivalency/blob/main/example.sh): two example commands to run experiments  
[main.py](https://github.com/CAHLR/skill-equivalency/blob/main/main.py): the main method to run experiments  
[arguments.py](https://github.com/CAHLR/skill-equivalency/blob/main/arguments.py): argument processing  
[translate.py](https://github.com/CAHLR/skill-equivalency/blob/main/translate.py): translation model learning  
[evaluate.py](https://github.com/CAHLR/skill-equivalency/blob/main/evaluate.py): evaluation logic  
[skill_representations](https://github.com/CAHLR/skill-equivalency/tree/main/skill_representations): six models to represent skills as vectors  
[data](https://github.com/CAHLR/skill-equivalency/tree/main/data): the datasets we release  
[output](https://github.com/CAHLR/skill-equivalency/tree/main/output): the output of two example experiments by running [example.sh](https://github.com/CAHLR/skill-equivalency/blob/main/example.sh)


## Datasets

We release some datasets we used in the paper at [data](https://github.com/CAHLR/skill-equivalency/tree/main/data) folder. They are either public or created by us.  
**Khan Academy**: exercise contents are web-scraped from [Khan Academy](https://www.khanacademy.org/) and its [github repo](https://github.com/Khan/khan-exercises).  
**ASSISTments**: clickstream data available on [ASSISTmentsData](https://sites.google.com/site/assistmentsdata/home/2012-13-school-data-with-affect), we further processed the clickstream data into skill sequences data.  
**Cognitive Tutor**: clickstream data available on [KDD Cup 2010](https://pslcdatashop.web.cmu.edu/KDDCup/downloads.jsp), we used Algebra I 2008-2009 and further processed the clickstream data into skill sequences data.  
**Skill equivalency labels**: we provide the equivalency labels between all three platforms annotated as described in the paper.


## Instructions

Run experiments by calling main.py and passing correct parameters, for example:
```
python main.py --src-name cog \
               --dst-name assist \
               --src-repr-model skill2vec \
               --dst-repr-model skill2vec \
               --labels-path ./data/labels_cog2assist.csv \
               --src-sequences-path ./data/sequences_cog.json \
               --src-skill2id-path ./data/skill2id_cog.json \
               --dst-sequences-path ./data/sequences_assist.json \
               --dst-skill2id-path ./data/skill2id_assist.json
```
For each run, `src-repr-model`, `dst-repr-model`, `labels-path` are required. Besides, depending on the models used for representation, different input data are required. If the model needs content (BOW, TFIDF, content2vec, content2vec_skill2vec, TAMF), `problems-path` should be given; and if the model involves context (skill2vec, content2vec_skill2vec, TAMF), `sequences-path` and `skill2id-path` should be specified.

You can also set various hyperparameters for skill representations and translation model. See [arguments.py](https://github.com/CAHLR/skill-equivalency/blob/main/arguments.py) for more details.


## Input data format

Please note that the library requires specific input data format. You can find examples in the [data](https://github.com/CAHLR/skill-equivalency/tree/main/data) folder.  
**Equivalency labels**: csv file with two columns named "source" and "destination" respectively  
**Problem contents**: tsv file with two columns named "skill" and "content". Each row is a problem, thus many rows can share the same the "skill".  
**Sequences**: two json files, a "skill2id" file that maps each skill to an int id, and a "sequences" file that contains the actual skill sequences. This "skill2id" is meant to prevent the "sequences" file from being too large by replacing the skill name strings with a smaller-sized id.  
