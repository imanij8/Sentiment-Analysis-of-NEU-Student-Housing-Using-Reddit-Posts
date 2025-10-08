from collections import Counter, defaultdict
import re
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
pd.set_option('future.no_silent_downcasting', True)
pio.renderers.default = 'browser'

class nlp:
    def __init__(self):
        self.data = defaultdict(dict)
        self.stop_words = set()

    def load_stop_words(self, stopfile):
        with open(stopfile, 'r') as f:
            self.stop_words = {line.strip().lower() for line in f if line.strip()}
        print("Loaded stop words:", self.stop_words)

    def simple_text_parser(self, filename):
        with open(filename, 'r') as f:
            text = f.read()
        raw_text = text
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        words = text.split()
        if self.stop_words:
            words = [word for word in words if word not in self.stop_words]
        results = {
            'wordcount': Counter(words),
            'numwords': len(words),
            'raw_text': raw_text
        }
        print("Parsed:", filename, ":", results)
        return results

    def load_text(self, filename, label=None, parser=None):
        if parser is None:
            results = self.simple_text_parser(filename)
        else:
            results = parser(filename)
        if label is None:
            label = filename
        for key, value in results.items():
            self.data[key][label] = value
        print(f"Loaded text for {label}: {results}")

    def wordcount_sankey(self, word_list=None, k=5):
        # Get document labels from the wordcount data
        doc_labels = list(self.data['wordcount'].keys())
        # If no custom word list is provided use the most common words
        if word_list is None:
            words_set = set()
            for doc in doc_labels:
                common_words = [word for word, _ in self.data['wordcount'][doc].most_common(k)]
                words_set.update(common_words)
            word_list = list(words_set)
        nodes = doc_labels + word_list
        doc_index = {doc: i for i, doc in enumerate(doc_labels)}
        word_index = {word: i + len(doc_labels) for i, word in enumerate(word_list)}

        sources, targets, values = [], [], []
        for doc in doc_labels:
            counter = self.data['wordcount'][doc]
            for word in word_list:
                count = counter.get(word, 0)
                if count > 1:
                    sources.append(doc_index[doc])
                    targets.append(word_index[word])
                    values.append(count)

        # Create the Sankey diagram
        sankey_fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=nodes,
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
            )
        )])

        sankey_fig.update_layout(title_text="Most Common Words Sankey", font_size=10)
        sankey_fig.show()

    def sentiment_breakdown(self):
        analyzer = SentimentIntensityAnalyzer()

        doc_labels = list(self.data['raw_text'].keys())
        n_docs = len(doc_labels)

        fig, axes = plt.subplots(n_docs, 1, figsize=(8, 4 * n_docs))
        if n_docs == 1:
            axes = [axes]

        for i, doc in enumerate(doc_labels):
            raw_text = self.data['raw_text'][doc]
            scores = analyzer.polarity_scores(raw_text)
            ax = axes[i]
            ax.bar(list(scores.keys()), list(scores.values()), color='skyblue')
            ax.set_title(f"Sentiment Breakdown for {doc}")
            ax.set_ylim(0, max(max(scores['neg'], scores['neu'], scores['pos']) * 1.1, 0.5))
        plt.tight_layout()
        plt.savefig('sentiment_breakdown.png')
        plt.show()

    def sentiment_heatmap(self):
        # heat map function
        analyzer = SentimentIntensityAnalyzer()

        doc_labels = list(self.data['raw_text'].keys())
        metrics = ['neg', 'neu', 'pos', 'compound']
        sentiment_matrix = []
        for metric in metrics:
            row = []
            for doc in doc_labels:
                raw_text = self.data['raw_text'][doc]
                scores = analyzer.polarity_scores(raw_text)
                row.append(scores[metric])
            sentiment_matrix.append(row)

        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(sentiment_matrix, cmap='coolwarm', aspect='auto')

        ax.set_xticks(range(len(doc_labels)))
        ax.set_xticklabels(doc_labels)
        ax.set_yticks(range(len(metrics)))
        ax.set_yticklabels(metrics)

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        for i in range(len(metrics)):
            for j in range(len(doc_labels)):
                ax.text(j, i, f"{sentiment_matrix[i][j]:.2f}",
                        ha="center", va="center", color="black")

        ax.set_title("Sentiment Heatmap (VADER Scores) per Document")
        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig('sentiment_heatmap.png')
        plt.show()



def main():
    nlpc = nlp()
    nlpc.load_stop_words('stopwords.txt')
    #nlpc.simple_text_parser('reddit_post1.json')
    #nlpc.load_text('reddit_post1.json')

    nlpc.load_text('reddit_post1.json', 'Post 1')
    nlpc.load_text('reddit_post2.json', 'Post 2')
    nlpc.load_text('reddit_post3.json', 'Post 3')
    nlpc.load_text('reddit_post4.json', 'Post 4')
    nlpc.load_text('reddit_post5.json', 'Post 5')
    nlpc.load_text('reddit_post6.json', 'Post 6')
    nlpc.load_text('reddit_post7.json', 'Post 7')
    nlpc.load_text('reddit_post8.json', 'Post 8')
    nlpc.load_text('reddit_post9.json', 'Post 9')
    nlpc.load_text('reddit_post10.json', 'Post 10')
    nlpc.load_text('reddit_post11.json', 'Post 11')
    nlpc.load_text('reddit_post12.json', 'Post 12')
    nlpc.load_text('reddit_post13.json', 'Post 13')
    nlpc.load_text('reddit_post14.json', 'Post 14')

    nlpc.wordcount_sankey(k=5)

    #sentiment plots
    #nlpc.sentiment_breakdown()

    #heatmap plot
    #nlpc.sentiment_heatmap()


if __name__ == '__main__':
    main()