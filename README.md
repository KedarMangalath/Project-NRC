# Project-NRC
This script describes a two-part emotion analysis system using a machine learning approach. It combines text processing, a lexicon-based method, and deep learning models for emotion classification and intensity prediction. Here’s a breakdown of the components and processes:

### Lexicon-based Sentiment Analysis:
- A lexicon (`nrc_lexicon`) is used to calculate scores for different emotions present in the text. It's based on the NRC Emotion Lexicon, a list of English words and their associations with eight basic emotions.
- For each tweet, emotion scores are computed by tallying the occurrences of each word that has an associated emotion in the lexicon.

### Preprocessing:
- Tweets are cleaned to remove URLs, user mentions, hashtags, and punctuation.
- The cleaned tweets are tokenized into words using NLTK's word tokenizer.

### Machine Learning Models:
- *Emotion Classification Model*: A deep learning model is designed to classify the emotions of tweets. The model architecture includes:
  - An `Input` layer that accepts preprocessed tweet sequences.
  - An `Embedding` layer to convert token indices into dense vectors of fixed size.
  - A `Bidirectional LSTM` layer that processes the embedded sequences. LSTM (Long Short-Term Memory) is adept at capturing long-range dependencies in sequence data, and the bidirectional wrapper allows it to consider both past and future context.
  - A `Dense` output layer with a `softmax` activation function to classify the tweet into one of the emotion categories.

- *Emotion Intensity Regression Model*: A separate model to predict the intensity of the emotions. Its architecture includes:
  - An `Input` layer that accepts the emotion scores from the NRC lexicon.
  - A `Dense` layer with `relu` activation to learn non-linear relationships.
  - An output `Dense` layer with a single neuron to predict the emotion intensity as a continuous value.

### Data and Training:
- Datasets for four different emotions (anger, fear, joy, sadness) are loaded and combined into a single dataset, which is then shuffled.
- Tokenization is performed on the combined dataset, followed by padding to ensure all sequences have the same length.
- Emotion labels are encoded using a label encoder and converted to categorical format for classification.
- The dataset is split into training and testing sets, with the split also applied to the emotion scores and intensity values.

### Model Training and Evaluation:
- The classification model is trained on the tokenized and padded tweet sequences, while the regression model is trained on the emotion scores derived from the NRC lexicon.
- After training, the classification model predicts the emotion category, and the regression model predicts the intensity of the emotions.
- Predictions are combined with the actual labels and intensities to form a results DataFrame for evaluation.
- Both models are evaluated using appropriate metrics: classification report for the classification model, and mean squared error (MSE), R-squared, mean absolute error (MAE), and Pearson correlation for the regression model.

### Summary:
This system uses a lexicon-based approach to inform a machine learning model, aiming to take advantage of the strengths of both methods. The classification model categorizes emotions, while the regression model predicts how strongly the emotion is expressed. The use of a Bidirectional LSTM enables the model to understand the context better, which is crucial for sentiment analysis.
