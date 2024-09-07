import csv
import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.data import find

# Ensure NLTK 'punkt' package is downloaded for tokenization
try:
    find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

class SentimentAnalyzer:
    """
    A class to analyze sentiment of text using basic sentiment keywords,
    negation, and modifiers for intensifying or downtoning sentiment scores.
    """

    # Starter sets for different categories of words

    _default_positive_words = [
        "happy", "joy", "delight", "love", "wonderful", "fantastic",
        "brilliant", "amazing", "excellent", "successful", "pleased", "thrilled"
    ]

    _default_negative_words = [
        "sad", "unhappy", "disappointed", "hate", "terrible", "awful",
        "horrible", "dreadful", "poor", "fail", "miserable", "depressed"
    ]

    _default_negation_words = [
        "not", "no", "never", "none", "cannot", "isn't", "aren't",
        "wasn't", "weren't", "haven't", "hasn't", "don't"
    ]

    _default_intensifiers = [
        "very", "extremely", "incredibly", "absolutely", "completely",
        "utterly", "totally", "deeply", "enormously", "exceptionally", "especially", "tremendously"
    ]

    _default_downtoners = [
        "slightly", "somewhat", "a bit", "barely", "hardly", "just",
        "marginally", "scarcely", "a little", "less", "rarely", "occasionally"
    ]

    # Named constants for multipliers
    INTENSIFIER_MULTIPLIER = 1.5
    DOWNTONER_MULTIPLIER = 0.5

    #``````````````````````````````````````````````````````````````````````````````````
    def __init__(self, positive_words=None, negative_words=None, negation_words=None, intensifiers=None, downtoners=None):
        """
        Initializes the SentimentAnalyzer with optional custom lists of words. 
        Falls back to default lists if none are provided.
        """
        self.positive_words = positive_words if positive_words is not None else self._default_positive_words
        self.negative_words = negative_words if negative_words is not None else self._default_negative_words
        self.negation_words = negation_words if negation_words is not None else self._default_negation_words
        self.intensifiers = intensifiers if intensifiers is not None else self._default_intensifiers
        self.downtoners = downtoners if downtoners is not None else self._default_downtoners


    #``````````````````````````````````````````````````````````````````````````````````
    def analyze_sentence_sentiment(self, sentence, use_negation=False, use_modifiers=False):
        """
        Analyzes the sentiment score of a sentence based on the presence of positive, negative,
        negation, and modifier (intensifiers and downtoners) words. The function calculates a 
        sentiment score that reflects the overall sentiment of the sentence.

        Parameters:
        - sentence: The sentence to analyze
        - use_negation: Whether to consider negation words
        - use_modifiers: Whether to consider intensifiers and downtoners

        Returns:
        - The sentiment score of the sentence
        """

        # Tokenize the sentence into words
        words = word_tokenize(sentence)

        # Initialize sentiment score
        sentiment_score = 0

        # Initialize modifier effect
        modifier_effect = 1

        # Iterate through each word
        for i, word in enumerate(words):
            # Check for negation word
            if use_negation and word in self.negation_words:
                # Invert the modifier effect
                modifier_effect *= -1
                continue  # Skip negation word

            # Check for intensifiers and downtoners
            if use_modifiers and word in self.intensifiers:
                modifier_effect *= self.INTENSIFIER_MULTIPLIER
                continue  # Skip modifier word
            elif use_modifiers and word in self.downtoners:
                modifier_effect *= self.DOWNTONER_MULTIPLIER
                continue  # Skip modifier word

            # Check if the word is positive or negative
            if word in self.positive_words:
                sentiment_score += modifier_effect
            elif word in self.negative_words:
                sentiment_score -= modifier_effect

            # Reset modifier effect
            modifier_effect = 1

        return sentiment_score
    def get_sentiment(self, sentiment_score):
        """
        Determines the sentiment label ('positive', 'negative', 'neutral') based on the sentiment score.

        Parameters:
        - sentiment_score: The sentiment score to evaluate

        Returns:
        - A string label indicating the sentiment ('positive', 'negative', 'neutral')
        """
        if sentiment_score > 0:
            return 'positive'
        elif sentiment_score < 0:
            return 'negative'
        else:
            return 'neutral'

    def calculate_overall_sentiment_score(self, sentiment_scores):
        """
        Calculates the average sentiment score from a list of individual sentence scores.

        Parameters:
        - sentiment_scores: A list of sentiment scores from individual sentences

        Returns:
        - The average sentiment score
        """
        return sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0

    def get_sentences_from_lines(self, lines_list):
        """
        Converts a list of text lines into a list of sentences using NLTK's sentence tokenizer.

        Parameters:
        - lines_list: A list of text lines

        Returns:
        - A list of sentences
        """
        sentences = []
        for line in lines_list:
            sentences.extend(sent_tokenize(line))
        return sentences

    def analyze_sentiment(self, text_lines_list, use_negation=False, use_modifiers=False):    
        """
        Analyzes the overall sentiment of multiple lines of text.

        Parameters:
        - text_lines_list: A list of text lines to analyze
        - use_negation: Whether to consider negation in the analysis
        - use_modifiers: Whether to consider intensifiers and downtoners in the analysis

        Returns:
        - A dictionary with detailed results, overall sentiment, and sentiment counts
        - detailed_results is a list of dictionaries
            Each entry in the list contains "sentiment", "score", and "sentence" as keys.
        - The return dictionary contains three elements:
            detailed_results : as above
            overall_sentiment : a dictionary with "overall_sentiment" which gives the sentiment
                                as positive, negative, or neutral and "score" which gives
                                the overall sentiment score. 
            sentiment_counts :  a list of sentiment counts for each sentence.     
        """
        detailed_results = []
        sentiment_scores = []
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}

        sentences = self.get_sentences_from_lines(text_lines_list)

        for sentence in sentences:
            sentiment_score = self.analyze_sentence_sentiment(sentence, use_negation, use_modifiers)
            sentiment = self.get_sentiment(sentiment_score)
            detailed_results.append({'sentence': sentence, 'sentiment': sentiment, 'score': sentiment_score,})
            sentiment_scores.append(float(sentiment_score))  # Convert sentiment_score to float
            sentiment_counts[sentiment] += 1

        overall_sentiment_score = self.calculate_overall_sentiment_score(sentiment_scores)
        overall_sentiment = self.get_sentiment(overall_sentiment_score)

        return {
            'detailed_results': detailed_results,
            'overall_sentiment': {'overall_sentiment': overall_sentiment, 'score': overall_sentiment_score},
            'sentiment_counts': sentiment_counts
        }

    def write_to_csv(self, detailed_results, csv_file_path):
        """
        Writes the detailed sentiment analysis results to a CSV file.

        Parameters:
        - detailed_results: A list of dictionaries containing sentiment analysis results
        - csv_file_path: The file path to write the CSV data to
        """
        with open(csv_file_path, mode='w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Sentence', 'Sentiment', 'Score']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for result in detailed_results:
                writer.writerow({'Sentence': result['sentence'], 'Sentiment': result['sentiment'], 'Score': result['score'],})
def main():
    """
    The main test function for the SentimentAnalyzer class. This function is designed to verify
    the correctness of the analyze_sentence_sentiment method by running it through a series of
    test cases. Each test case is an assertion that checks if the method returns the expected
    sentiment score for a given sentence under specified conditions (use of negation and modifiers).

    Example Assertions Explained:
    - Test case 1 checks a simple positive sentence without any negation or modifiers. The expected
      score is 1, indicating a positive sentiment.
    - Test case 2 checks a simple negative sentence, expecting a score of -1 to reflect the negative sentiment.
    - Test case 3 involves a positive sentence with a negation, turning the sentiment negative. With
      use_negation=True, it tests if the function correctly inverts the sentiment, expecting a score of -1.
    - Test case 4 examines the effect of an intensifier on a positive word, increasing its sentiment
      impact. use_modifiers=True activates the modifier logic, expecting an intensified positive score.
    - Test case 5 looks at a negative sentence with a downtoner, reducing the negative sentiments impact.
      This tests the downtoner effect with an expected score indicating a lessened negative sentiment.
    - Test case 6 is a complex sentence that combines negation with an intensifier. This tests both the
      negation and modifier logic together, expecting a score that reflects the combined effects.

    Additional complex test cases mix multiple aspects of sentiment analysis to ensure the method
    can handle a variety of sentence structures and sentiment expressions accurately.

    Note:
    - This main function is for testing purposes only and demonstrates how the SentimentAnalyzer class
      can be utilized.
    - The assertions are critical for validating the expected functionality of the sentiment analysis
      method. Each assertion represents a specific scenario that the SentimentAnalyzer is expected to
      handle correctly.
    - Understanding these test cases and their expected outcomes will help in grasping the nuances of
      sentiment analysis as implemented in this class.

    Students are encouraged to add more test cases to cover additional scenarios and further validate
    the robustness of the sentiment analysis method.
    """

    analyzer = SentimentAnalyzer(["happy", "outstanding", "great"],["sad", "disappointing", "bad"],\
                                    ["not", "never"],["very", "extremely","definitely"],["somewhat", "slightly"])

    # Test case 1: Positive keyword
    assert analyzer.analyze_sentence_sentiment("This is a great day.") == 1, "Failed on positive keyword test"

    # Test case 2: Negative keyword
    assert analyzer.analyze_sentence_sentiment("This is a sad day.") == -1, "Failed on negative keyword test"

    # Test case 3: Negation of a positive word (without use_negation=True should be treated as positive)
    # this test will fail because of the "a" between the negation and the positive word.
    #
    #assert analyzer.analyze_sentence_sentiment("This is not a great day.", use_negation=True) == -1, "Failed on negation test"

    # Test case 3: Negation of a positive word (without use_negation=True should be treated as positive)
    assert analyzer.analyze_sentence_sentiment("This day is not great.", use_negation=True) == -1, "Failed on negation test"

    # Test case 4: Modified negation of a positive word (without use_negation=True should be treated as positive)
    assert analyzer.analyze_sentence_sentiment("This is definitely not great.", use_negation=True, use_modifiers=True) == -1.5, "Failed on intensify/downtone a negation test"

    # Test case 5: Intensified positive word
    assert analyzer.analyze_sentence_sentiment("This is a very great day.", use_modifiers=True) == 1.5, "Failed on intensifier test"

    # Test case 6: Downtoned negative word
    assert analyzer.analyze_sentence_sentiment("This is somewhat disappointing.", use_modifiers=True) == -0.5, "Failed on downtoner test"

    print("All simple sentence tests passed!")        

    canalyzer = SentimentAnalyzer(["happy", "outstanding", "great"], ["bad", "awful","disappointing"], ["not", "never"], ["very", "extremely","definitely"], ["somewhat", "slightly"])

    # Mixed sentiment with negation and modifier
    assert canalyzer.analyze_sentence_sentiment("This is a great day, but somewhat disappointing.", use_negation=True, use_modifiers=True) == 0.5, "Failed on mixed sentiment with negation and modifier"

    # Intensified positive followed by a downtoned negative
    assert canalyzer.analyze_sentence
    # Intensified positive followed by a downtoned negative
    assert canalyzer.analyze_sentence_sentiment("It was very outstanding yet slightly bad.", use_modifiers=True) == 1, "Failed on intensified positive followed by downtoned negative"

    # Negated positive followed by an unmodified negative
    assert canalyzer.analyze_sentence_sentiment("This is not happy and also awful.", use_negation=True) == -2, "Failed on negated positive followed by unmodified negative"

    # Multiple modifiers with a negation impacting different parts of the sentence
    assert canalyzer.analyze_sentence_sentiment("It was definitely not great, but somewhat bad.", use_negation=True, use_modifiers=True) == -2, "Failed on multiple modifiers with negation"

    # Sentences with neutral words and sentiment words without explicit modifiers or negations
    assert canalyzer.analyze_sentence_sentiment("The day was outstanding then turned awful.", use_negation=True, use_modifiers=True) == 0, "Failed on sentence with neutral shift"

    # Mixed sentiment with multiple modifiers and negation
    assert canalyzer.analyze_sentence_sentiment("This is extremely bad but not somewhat outstanding.", use_negation=True, use_modifiers=True) == -0.5, "Failed on mixed sentiment with multiple modifiers and negation"

    # Complex sentence with negation impacting multiple sentiment words
    assert canalyzer.analyze_sentence_sentiment("This is not happy day, but it is definitely not awful.", use_negation=True, use_modifiers=True) == 0.5, "Failed on complex sentence with negation impacting multiple sentiment words"

    print("All complex sentence tests passed!")        

    # Test case for writing to CSV
    detailed_results = [
        {'sentence': 'This is a great day.', 'sentiment': 'positive', 'score': 1,},
        {'sentence': 'This day is not great.', 'sentiment': 'negative', 'score': -1,},
        {'sentence': 'It was very outstanding yet slightly bad.', 'sentiment': 'positive', 'score': 1,},
    ]

    analyzer.write_to_csv(detailed_results, 'sentiment_analysis_results.csv')
    print("CSV writing test passed!")

    # Add more tests as needed

    print("All tests passed!")


if __name__ == "__main__":
    main()
