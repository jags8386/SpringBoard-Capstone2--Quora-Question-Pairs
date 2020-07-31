import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer


nlp = spacy.load('en_core_web_sm')

def create_features(df, word2tfidf):
    vecs = []
    for qu1 in tqdm(list(df)):
    
        doc1 = nlp(qu1) 
        #creating object of   GLOVE model  so that we can get vetor representation of our words
    
        # Creating a matrix of N x M where N is is number of word is given line and M i.e. 96 which is the vector representaion of 1st word
        mean_vec = np.zeros([len(doc1), len(doc1[0].vector)])
    
        # Looping to all words in the given sentence 
    
        for word1 in doc1:
        
       # word2vec ( Creating Vector Representation of every word ) which is 96
    
            vec = word1.vector
        
            # Using try and catch to prefent key error [ For the words that are not there in our word2tfidf dict like empty space ]
        
            try:
                idf = word2tfidf[str(word1)]
            except:
                idf = 0
        
        # adding up all the words generated in the matrix (word2vec matrix * the word2tfidf Corresponding to that word)
        mean_vec += vec * idf
        mean_vec = mean_vec.mean(axis=0)
        
        vecs.append(mean_vec)  # Storing the vector representation of every sentence into an array
    return list(vecs)
    
    

# =============================================================================
# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
# =============================================================================
def main(input_filepath, output_filepath):
    df_data = pd.read_csv(input_filepath)
    
    df=df_data[['cleanQ1','cleanQ2','is_duplicate']]
    df['cleanQ1'] = df['cleanQ1'].apply(lambda x: str(x))
    df['cleanQ2'] = df['cleanQ2'].apply(lambda x: str(x))
    
    # merging questions of both Q1 and Q2 to a single list in which first 
    #404287 index will be of question 1 and then rest of question 2
    questions = list(df['cleanQ1']) + list(df['cleanQ2'])  

    tfidf = TfidfVectorizer() #  Convert a collection of raw documents to a matrix of TF-IDF features

    tfidf.fit_transform(questions)  # Converting out text to a matrix of TF-IDF features

    # mapping our feature_names with threre resptive tf-idf score  (
    #dict key:word and value:tf-idf score )
    word2tfidf = dict(zip(tfidf.get_feature_names(), tfidf.idf_))
    # en_vectors_web_lg, which includes over 1 million unique vectors.
    df['q1_feats_m'] = create_features(df['cleanQ1'], word2tfidf)
    df['q2_feats_m'] = create_features(df['cleanQ2'], word2tfidf)
    
    df1 = df_data.drop(['cleanQ1', 'cleanQ2'], axis=1)
    df3 = df.drop(['cleanQ1', 'cleanQ2', 'is_duplicate'],axis=1)
    df3_q1 = pd.DataFrame(df3.q1_feats_m.values.tolist(), index= df3.index)
    df3_q2 = pd.DataFrame(df3.q2_feats_m.values.tolist(), index= df3.index)
    df3_q1['id']=df1['id']
    df3_q2['id']=df1['id']
    #df1  = df1.merge(df2, on='id',how='left')
    df2  = df3_q1.merge(df3_q2, on='id',how='left')
    result  = df1.merge(df2, on='id',how='left')
    result.to_csv(output_filepath, index=False)
   


    
    """ Runs data processing scripts to turn clean data from (../processed) 
    into featured data ready to be analyzed (saved in ../processed).
    """



if __name__ == '__main__':
    
    p = Path(__file__).parents[2]
    
    '''log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
   # load_dotenv(find_dotenv())'''

    main(str(p)+"/data/processed/clean_data_code.csv", 
         str(p)+"/data/processed/featured.csv")