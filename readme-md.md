# Sentiment Analysis Project

This project implements a simple sentiment analysis system using Python. It combines both machine learning (Naive Bayes) and rule-based (TextBlob) approaches to analyze the sentiment of text.

## Features

- Text preprocessing and cleaning
- Sentiment analysis using TextBlob
- Machine learning-based classification using Naive Bayes
- Support for both single text analysis and batch processing
- Easy-to-use Python interface

## Requirements

```
numpy
pandas
scikit-learn
textblob
nltk
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/[your-username]/sentiment-analysis.git
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

```python
from sentiment_analyzer import SentimentAnalyzer

# Initialize the analyzer
analyzer = SentimentAnalyzer()

# Analyze single text
text = "This product is amazing!"
result = analyzer.analyze_sentiment(text)
print(f"Sentiment: {result}")
```

## Project Structure

```
sentiment-analysis/
├── sentiment_analyzer.py   # Main implementation
├── requirements.txt        # Project dependencies
└── README.md              # Project documentation
```

## Future Improvements

- Add support for more languages
- Implement deep learning models
- Add API interface
- Improve accuracy with larger training dataset
- Add cross-validation

## Contributing

Feel free to open issues and pull requests to improve the project.

## License

MIT License
