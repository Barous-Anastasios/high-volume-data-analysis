import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

def generate_wordcloud(words, category):
    # The matplotlib way:
    word_cloud = WordCloud(width = 512, height = 512, background_color='black', stopwords=STOPWORDS).generate(words)
    plt.figure(figsize=(10,8),facecolor = 'white', edgecolor='blue')
    plt.imshow(word_cloud)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig('./../dist/wordcloud/matplotlib/'+category+'.jpg')
    # plt.show()

    # The pil way
    image = word_cloud.to_image()
    word_cloud.to_file('./../dist/wordcloud/images/'+category+'.png')
    # image.show()

df = pd.read_csv('./../data/train_set.csv', delimiter="\t", encoding='utf-8')
categories = ['Football', 'Film', 'Technology', 'Politics', 'Business']

for category in categories:
    words = ''
    articles = df[df["Category"] == category]
    first_ten = articles.head(20)
    for rowInd, row in first_ten.iterrows():
        words += row['Title']
    generate_wordcloud(words, category)