# -*- coding: utf-8 -*-
"""Sentiment Analysis.ipynb

Original file is located at
    https://colab.research.google.com/drive/1b9exXv5TL_Tnd8ZHr6JwMSikxbCH3Gui
"""


"""

## Install Libraries and Download NLTK Data, 'vader_lexicon' data required by `SentimentIntensityAnalyzer`.

"""

get_ipython().system('pip install nltk')

import nltk
nltk.download('vader_lexicon')


"""
## Mount Google Drive
## Specify Spreadsheet File Path and Column
"""

from google.colab import drive
drive.mount('/content/drive')


spreadsheet_file_path = '/content/drive/MyDrive/Colab Notebooks/Cmt sentiment update.csv' # @param {type:"string"}
comments_column_name = 'ROUND LAB' # @param {type:"string"}

print(f"Spreadsheet file path set to: {spreadsheet_file_path}")
print(f"Comments column name set to: {comments_column_name}")

"""## Load Comments from Spreadsheet

### Subtask:
Use the `pandas` library to read the spreadsheet file from the specified Google Drive path and extract the comments from the designated column into a Python list.
"""
import pandas as pd
import os

# Determine file type and read the spreadsheet
file_extension = os.path.splitext(spreadsheet_file_path)[1].lower()

if file_extension == '.csv':
    df = pd.read_csv(spreadsheet_file_path)
    print(f"Successfully loaded CSV file: {spreadsheet_file_path}")
elif file_extension in ('.xls', '.xlsx'):
    df = pd.read_excel(spreadsheet_file_path)
    print(f"Successfully loaded Excel file: {spreadsheet_file_path}")
else:
    raise ValueError("Unsupported file type. Please provide a .csv or .xlsx file.")

# Extract comments into a Python list
if comments_column_name in df.columns:
    comments = df[comments_column_name].dropna().tolist()
    print(f"Extracted {len(comments)} comments from column '{comments_column_name}'.")
else:
    raise ValueError(f"Column '{comments_column_name}' not found in the spreadsheet. Available columns: {df.columns.tolist()}")

print("First 5 comments:")
for i, comment in enumerate(comments[:5]):
    print(f"- {comment}")

"""
Initialize the VADER sentiment analyzer and define a function to classify the sentiment of a comment as positive, negative, or neutral based on its compound score. 
This function will be used in the next step to classify all comments.

"""

from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize VADER sentiment intensity analyzer
sid = SentimentIntensityAnalyzer()

# Define a function to get sentiment label from VADER scores
def get_vader_sentiment(text):
    scores = sid.polarity_scores(text)
    if scores['compound'] >= 0.05:
        return 'Positive'
    elif scores['compound'] <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

print("VADER SentimentIntensityAnalyzer initialized and sentiment classification function defined.")

"""
Iterate through the loaded `comments` list, apply the `get_vader_sentiment` function to each comment, and store the original comment along with its classified sentiment. 
This fulfills the subtask of classifying sentiment for each comment using VADER.

# Task
## Customize VADER Lexicon
Add or edit specific Vietnamese words and phrases (e.g., "thích", "mê", "ưng", "chân ái", "xịn", "ổn") to the VADER lexicon with appropriate positive sentiment scores to ensure accurate classification for these terms.
"""

# 1. Initialize SentimentIntensityAnalyzer to get the default lexicon
default_sid = SentimentIntensityAnalyzer()
default_vader_lexicon = default_sid.lexicon.copy()

# 2. Create a dictionary for custom Vietnamese sentiment words
# Assigning positive sentiment scores (between -4.0 and +4.0, similar to VADER's range)
vietnamese_custom_lexicon = {
    'thích': 1.5,        # like, enjoy
    'mê': 2.0,           # adore, really like
    'ưng': 1.8,          # satisfy, approve
    'chân ái': 2.8,      # true love, soulmate (strong positive)
    'tốt': 1.8,          # good
    'đỉnh': 2.0,         # peak, excellent
    'xịn': 2.2,          # excellent, high quality
    'ổn': 1.0,           # good, fine, okay
    'oki': 1.0,          # okay
    'okee': 1.0,         # okay
    'yêu': 2.0,          # love
    'tuyệt vời': 2.8,    # wonderful, excellent
    'fan': 0,            # vague
    'không thích': -2.8, # dislike (negative)
    'không mê': -1.5,    # don't adore (negative)
    'không ưng': -1.8,   # not satisfy (negative)
    'k thích': -2.8,     # shorthand for 'không thích'
    'k mê': -1.5,        # shorthand for 'không mê'
    'k ưng': -1.8,       # shorthand for 'không ưng'
    'nhạt': -1.8,        # bland, dull (negative)
    'kích ứng': -2.0,    # irritation (negative)
    'dị ứng': -2.0,      # allergy (negative)
    'chê': -2.5,         # criticize, complain (negative)
    'đắt': -1.5,        # expensive (negative)
    'mắc': -1.5,         # expensive (negative)
    'không ổn': -1.0,    # not good (negative)
    'k có': -1.0,
    'k đủ': -1.0,        # don't have (can imply negative if context specific)
    'tệ': -2.0,          # bad
    'dở': -1.5           # poor quality

}

# 3. Merge the custom lexicon with the default VADER lexicon
# Custom scores will override existing ones
custom_vader_lexicon = default_vader_lexicon.copy()
custom_vader_lexicon.update(vietnamese_custom_lexicon)

# 4. Re-initialize SentimentIntensityAnalyzer with the customized lexicon
sid = SentimentIntensityAnalyzer()
sid.lexicon = custom_vader_lexicon

print("VADER lexicon customized with Vietnamese words and SentimentIntensityAnalyzer re-initialized.")

"""
re-run the sentiment classification on the `comments` list using this updated `sid` to ensure the custom lexicon is applied.

"""

vader_classified_comments_custom = []

print(f"Classifying {len(comments)} comments using customized VADER lexicon...")

for i, comment in enumerate(comments):
    if pd.isna(comment): # Skip NaN values
        continue
    sentiment = get_vader_sentiment(str(comment))
    vader_classified_comments_custom.append({
        'Original Comment': str(comment),
        'VADER Sentiment (Custom)': sentiment
    })
    if (i + 1) % 10 == 0:
        print(f"  Classified {i + 1}/{len(comments)} comments.")

print(f"\nFinished classifying {len(vader_classified_comments_custom)} comments.")
print("\nFirst 5 VADER classified comments with custom lexicon:")
for item in vader_classified_comments_custom[:5]:
    print(f"- Comment: '{item['Original Comment']}', Sentiment: {item['VADER Sentiment (Custom)']}")

"""
Organize these results into a pandas DataFrame and display it as a clean, readable table.

"""

import pandas as pd

# Create a DataFrame from the vader_classified_comments_custom list
vader_custom_results_df = pd.DataFrame(vader_classified_comments_custom)

# Display the DataFrame
print("VADER Sentiment Analysis Results (Custom Lexicon):")
# Set display options to show all columns and more rows
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
# Using to_string() for better display in Colab for large DataFrames
print(vader_custom_results_df.to_string())

"""
## Export Results to CSV

"""

output_csv_path = '/content/drive/MyDrive/Colab Notebooks/roundlab_sentiment.csv' # @param {type:"string"}

vader_custom_results_df.to_csv(output_csv_path, index=False)

print(f"Sentiment analysis results saved to: {output_csv_path}")

"""## Summarize VADER Sentiment Analysis Results

### Subtask:
Calculate and display the count of comments for each sentiment category (Positive, Neutral, Negative) from the classified results, and visualize these counts using a bar chart.

"""

import matplotlib.pyplot as plt
import seaborn as sns

# Calculate the value counts for each sentiment category
sentiment_counts = vader_custom_results_df['VADER Sentiment (Custom)'].value_counts()

print("Sentiment Distribution (Custom VADER Lexicon):")
print(sentiment_counts)

# Visualize the sentiment distribution
plt.figure(figsize=(8, 6))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette='viridis')
plt.title('Distribution of Sentiments (Custom VADER Lexicon)')
plt.xlabel('Sentiment')
plt.ylabel('Number of Comments')
plt.show()



"""# Task
Generate a word cloud visualization and summarize key words from the comments in the 'TORRIDEN' column of the spreadsheet file `/content/drive/MyDrive/Colab Notebooks/Cmt sentiment.csv`.
"""
from google.colab import drive
drive.mount('/content/drive')


spreadsheet_file_path = '/content/drive/MyDrive/Colab Notebooks/Cmt sentiment.csv' # @param {type:"string"}
comments_column_name = 'KLAIRS' # @param {type:"string"}

print(f"Spreadsheet file path set to: {spreadsheet_file_path}")
print(f"Comments column name set to: {comments_column_name}")


import pandas as pd
import os

# Determine file type and read the spreadsheet
file_extension = os.path.splitext(spreadsheet_file_path)[1].lower()

if file_extension == '.csv':
    df = pd.read_csv(spreadsheet_file_path)
    print(f"Successfully loaded CSV file: {spreadsheet_file_path}")
elif file_extension in ('.xls', '.xlsx'):
    df = pd.read_excel(spreadsheet_file_path)
    print(f"Successfully loaded Excel file: {spreadsheet_file_path}")
else:
    raise ValueError("Unsupported file type. Please provide a .csv or .xlsx file.")

# Extract comments into a Python list
if comments_column_name in df.columns:
    comments = df[comments_column_name].dropna().tolist()
    print(f"Extracted {len(comments)} comments from column '{comments_column_name}'.")
else:
    raise ValueError(f"Column '{comments_column_name}' not found in the spreadsheet. Available columns: {df.columns.tolist()}")

print("\nFirst 5 comments:")
for i, comment in enumerate(comments[:5]):
    print(f"- {comment}")

"""
Preprocess the text: tokenizing the words, removing punctuation, converting text to lowercase, and eliminating common Vietnamese stopwords to focus on meaningful terms. 

"""

get_ipython().system('pip install underthesea')

import re
from underthesea import word_tokenize

# Define a list of Vietnamese stopwords (can be expanded)
vietnamese_stopwords = [
    'là', 'và', 'của', 'có', 'không', 'được', 'trên', 'với', 'trong', 'ra',
    'tôi', 'em', 'anh', 'chị', 'bạn', 'mình', 'nó', 'họ', 'cô', 'dì',
    'cái', 'những', 'một', 'hai', 'ba', 'bốn', 'năm', 'sáu', 'bảy', 'tám', 'chín', 'mười',
    'đã', 'sẽ', 'đang', 'vẫn', 'cũng', 'lại', 'rồi', 'thì', 'mà', 'nữa',
    'như', 'khi', 'nếu', 'tuy', 'vì', 'do', 'để', 'hay', 'hoặc', 'và',
    'nhưng', 'còn', 'muốn', 'phải', 'nên', 'thật', 'rất', 'quá', 'hơn', 'kém',
    'biết', 'thấy', 'nghe', 'nói', 'làm', 'đi', 'đến', 'về', 'ở', 'từ',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
    'ạ', 'ơi', 'à', 'ừ', 'nhé', 'nhau', 'hả', 'gì', 'chứ', 'đâu', 'nào', 'mà',
    'e', 'c', 'a', 'b', 'd', 'k', 'n', 'm', 'p', 'q', 'r', 's', 't', 'u', 'v', 'x', 'y', 'z'
]

def preprocess_text(text):
    text = str(text).lower()  # Convert to lowercase
    text = re.sub(r'[^À-ỹa-z0-9\s]', '', text) # Remove punctuation and special characters, keep Vietnamese characters
    tokens = word_tokenize(text, format='text').split() # Tokenize using underthesea and split into words
    filtered_tokens = [word for word in tokens if word not in vietnamese_stopwords and len(word) > 1]
    return ' '.join(filtered_tokens)

# Apply preprocessing to all comments
processed_comments = [preprocess_text(comment) for comment in comments]

print(f"Finished preprocessing {len(processed_comments)} comments.")
print("First 5 processed comments:")
for i, comment in enumerate(processed_comments[:5]):
    print(f"- {comment}")

"""
Combine all processed comments into a single string and then use the `WordCloud` library to create and display the visualization. 

"""

get_ipython().system('pip install wordcloud')


from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Combine all processed comments into a single string
all_words = ' '.join(processed_comments)

# Generate word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white', collocations=False).generate(all_words)

# Display the generated image:
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Comments')
plt.show()

print("Word cloud generated and displayed.")

"""
Extract the top N most frequent words from the processed comments using `nltk.FreqDist` (or `collections.Counter`) and display them.


"""

from collections import Counter
import nltk

# Split the combined string of processed words into individual words
words = all_words.split()

# Calculate word frequency
word_freq = Counter(words)

# Get the top N most common words
top_n_words = 10  # You can adjust this number
most_common_words = word_freq.most_common(top_n_words)

print(f"\nTop {top_n_words} most common words in comments:")
for word, count in most_common_words:
    print(f"- {word}: {count}")

"""
### Subtask:
Provide a comprehensive summary of the generated word cloud and the key words identified from the comments.

### Data Analysis Key Findings
*   The spreadsheet `/content/drive/MyDrive/Colab Notebooks/Cmt sentiment.csv` was successfully loaded, and 240 comments were extracted from the 'TORRIDEN' column.
*   Comments were preprocessed by converting to lowercase, removing punctuation, tokenizing using `underthesea`, and filtering out a list of Vietnamese stopwords.
*   A word cloud visualization was successfully generated and displayed, providing a visual representation of the most frequent words.
*   The top 10 most common words identified from the comments are: 'uống' (56 occurrences), 'iso' (28 occurrences), 'mụn' (24 occurrences), 'khô' (23 occurrences), 'bị' (19 occurrences), 'luôn' (18 occurrences), 'dùng' (18 occurrences), 'da' (17 occurrences), 'bà' (17 occurrences), and 'môi' (16 occurrences).

### Insights or Next Steps
*   The high frequency of words like 'mụn' (acne), 'khô' (dry), 'dùng' (use), 'da' (skin), and 'môi' (lips) suggests that the comments are likely related to skincare products, possibly addressing issues like acne or dry skin/lips.
*   Further investigation into the context of words like 'uống' (drink) and 'iso' is recommended to understand their specific relevance within product comments, as they might indicate unique product forms or ingredients.
"""
