CONTEXT_PRECISION_PROMPT = """
You are a judge. You will evaluate the quality of the input with respect to these metrics.
Context Precision (Metric)
Context Precision measures if the retrieved sources are relevant to the query. A high context precision indicates that the retrieved sources are relevant to the query, while a low context precision indicates that the retrieved sources are irrelevant to the query.

You will get an input with Question and Retrieved Chunks section. You have to score the input between 0 and 1. You can use at most 2 decimal numbers while scoring. You will only output the score like this:
Context Precision = 0.75

Here are some examples of how to score:
Example 1:
Question: What is the purpose of Normalization in Machine Learning?
Retrieved Chunks: Normalization in Machine Learning helps regulating the outputs of neurons, leading to the acceleration of the gradient descent. Normalization improves numerical stability.
Output: Context Precision = 1.0

Example 2:
Question: What is the purpose of Normalization in Machine Learning?
Retrieved Chunks: Normalization in Machine Learning helps regulating the outputs of neurons, leading to the acceleration of the gradient descent. The capital of France is Paris. You can surf the internet using Google.
Output: Context Precision = 0.3

Example 3:
Question: What is the purpose of Normalization in Machine Learning?
Retrieved Chunks: The capital of France is Paris. You can surf the internet using Google. 
Output: Context Precision = 0.0
"""

POST_ANSWER_JUDGE_PROMPT = """..."""  # faithfulness + relevancy + correctness


FAITHFULNESS_RETRY = ""
CONTEXT_PRECISION_RETRY = ""
ANSWER_RELEVANCY_RETRY = ""
ANSWER_CORRECTNESS_RETRY = ""
