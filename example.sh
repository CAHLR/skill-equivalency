python main.py --src-name cog \
               --dst-name assist \
               --src-repr-model skill2vec \
               --dst-repr-model skill2vec \
               --labels-path ./data/labels_cog2assist.csv \
               --src-sequences-path ./data/sequences_cog.json \
               --src-skill2id-path ./data/skill2id_cog.json \
               --dst-sequences-path ./data/sequences_assist.json \
               --dst-skill2id-path ./data/skill2id_assist.json \
               
python main.py --src-name khan \
               --dst-name assist \
               --src-repr-model content2vec \
               --dst-repr-model skill2vec \
               --labels-path ./data/labels_khan2assist.csv \
               --src-problems-path ./data/problems_khan.tsv \
               --dst-sequences-path ./data/sequences_assist.json \
               --dst-skill2id-path ./data/skill2id_assist.json \
