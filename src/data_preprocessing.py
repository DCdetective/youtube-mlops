import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

nltk.download('stopwords')
nltk.download('punkt')


log_dir = 'logs'
os.makedirs(log_dir,exist_ok=True)

logger = logging.getLogger('data preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir , 'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def transform_text(text):
    """
    Transforms the input text by converting it to lowercase, tokenizing, removing stopwords and punctuation, and stemming.
    """
    ps = PorterStemmer()
    # Convert to lowercase
    text = text.lower()
    # Tokenize the text
    text = nltk.word_tokenize(text)
    # Remove non-alphanumeric tokens
    text = [word for word in text if word.isalnum()]
    # Remove stopwords and punctuation
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    # Stem the words
    text = [ps.stem(word) for word in text]
    # Join the tokens back into a single string
    return " ".join(text)

def preprocess_df(df , text_column ='text' ,target_column = 'target'):
    try:
        logger.debug("Starting preprocessing for DataFrame")
        encoder = LabelEncoder()
        df['target'] = encoder.fit_transform(df['target'])
        logger.debug("Target column encoded")
        df = df.drop_duplicates(keep = 'first')
        logger.debug("Duplicates Removed")
        
        df.loc[: , text_column] = df[text_column].apply(transform_text)
        logger.debug('Text column transformed')
        return df
    except KeyError as e:
        logger.error('Column not found: %s', e)
        raise
    except Exception as e:
        logger.error('Error during text normalization: %s', e)
        raise
    
    
def main():
    try:
        train_data = pd.read_csv("./data/raw/train.csv")
        test_data = pd.read_csv("./data/raw/test.csv")
        logger.debug("Data loaded properly")
        
        train_processed_data = preprocess_df(train_data)
        test_processed_data = preprocess_df(test_data)
        
        data_path = os.path.join("./data" , "interim")
        os.makedirs(data_path,exist_ok=True)
        
        train_processed_data.to_csv(os.path.join(data_path , "train_processed.csv"))
        test_processed_data.to_csv(os.path.join(data_path , 'test_processed.csv'))
        logger.debug('Processed data saved to %s', data_path)
    except FileNotFoundError as e:
        logger.error('File not found: %s', e)
    except pd.errors.EmptyDataError as e:
        logger.error('No data: %s', e)
    except Exception as e:
        logger.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")
        
if __name__ == '__main__':
    main()