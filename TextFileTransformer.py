from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from bert_score_eval import get_bertscore
from perpexlity_score import perplexity_score
from absa import Absa

class TextFileTransformer(BaseEstimator, TransformerMixin):
    '''
    Input - Text file of conversation with separator of #### between each dialogue
    Output - Pandas Dataframe with metric 
    '''

    def __init__(self, file_path):

        self.file_path = file_path
        self.data = None

    def fit(self, X, y=None):
        # The fit method is typically used for parameter tuning in transformers.
        return self

    def transform(self,X):
        print("1. Reading File into Dataframe")
        self.data=self.read_file()
        print("2. Calculating pairwise Bert Metrics")
        self.data=self.calculate_pairwise_bert_score()

        print("Calculate bert score wrt seed prompt")
        self.data=self.calculate_seed_bert_score()

        print("3. Calculating Perplexity Score Metrics")
        self.data=self.calculate_perplexity_score()


        print("4. Calculating aspect based Sentiments Metrics")
        absa=Absa(self.data)
        self.data=absa.get_absa()
        return self.data

    def read_file(self):
        """
        Input: File with conversations separated by #####
        Output : Dataframe with character and dialogue    
        """
        try:
            with open(self.file_path, 'r') as file:
                # Read the entire file into a string
                file_content = file.read()

                # Separate the file into chunks using the keyword "#####"
                separated_chunks = file_content.split('#####')

                # Remove whitespaces and new line characters from each element
                cleaned_chunks = [chunk.strip() for chunk in separated_chunks]

                # Create a list to store character names and dialogues
                data = []

                # Iterate through the cleaned chunks to extract character names and dialogues
                for i in range(0, len(cleaned_chunks), 2):
                    character_name = cleaned_chunks[i]
                    dialogue = cleaned_chunks[i + 1] if i + 1 < len(cleaned_chunks) else ""  # Handle odd-length chunks

                    data.append({'Character': character_name, 'Dialogue': dialogue})

                # Convert the list of dictionaries to a DataFrame
                df = pd.DataFrame(data)
                return df

        except FileNotFoundError:
            print(f"Error: File '{self.file_path}' not found.")
            return None
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def calculate_perplexity_score(self):
        px=perplexity_score()
        self.data['Perplexity']=self.data['Dialogue'].apply(px.calculate)
        return self.data
    
    def calculate_pairwise_bert_score(self):
        # Create new columns to store similarity scores
        self.data['F1_Score'] = 0.0

        # Iterate through each pair of adjacent dialogues to calculate and store similarity scores
        for i in range(1, len(self.data)):  # Iterate up to the second-to-last row
            candidate_dialogue = self.data.at[i, 'Dialogue']
            reference_dialogue = self.data.at[i -1, 'Dialogue']

            # Calculate similarity scores
            precision, recall, f1_score = get_bertscore(candidate_dialogue, reference_dialogue)
            #print(precision,recall,f1_score)

            # Store the scores in the DataFrame
            self.data.loc[self.data.index[i], 'F1_Score'] = f1_score
        return self.data
    
    def calculate_seed_bert_score(self):

        self.data['F1_Score_Seed'] = 0.0

        # Iterate through each pair of adjacent dialogues to calculate and store similarity scores
        for i in range(3, len(self.data)):  # Iterate up to the second-to-last row
            if i%2 == 1:
                candidate_dialogue = self.data.at[i, 'Dialogue']
                reference_dialogue = self.data.at[1, 'Dialogue']
            else:
                candidate_dialogue = self.data.at[i, 'Dialogue']
                reference_dialogue = self.data.at[2, 'Dialogue']

            # Calculate similarity scores
            precision, recall, f1_score = get_bertscore(candidate_dialogue, reference_dialogue)
            #print(precision,recall,f1_score)

            # Store the scores in the DataFrame
            self.data.loc[self.data.index[i], 'F1_Score_Seed'] = f1_score
        return self.data


# # Example usage:
# file_path = 'your_file.txt'  # Replace with the actual file path
# text_transformer = TextFileTransformer(file_path)

# # Transform the file
# df_result = text_transformer.transform(None)

# # Display the resulting DataFrame
# print(df_result)