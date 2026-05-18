CONTEXT_PRECISION_PROMPT = """
You are a judge. You will evaluate the quality of the input with respect to this metric.

Context Precision (Metric)
Context Precision measures if the retrieved sources are relevant to the query. 
A high context precision indicates that the retrieved sources are relevant to the query,
while a low context precision indicates that the retrieved sources are irrelevant to the query.

You will get an input with Question and Retrieved Chunks section. You have to score the input between 0 and 1. You can use at most 2 decimal numbers while scoring. You will only output the score like this:
Context Precision = 0.75

Here are some examples of how to score:
Example 1:
Question: What is the purpose of Normalization in Machine Learning?
Retrieved Chunks: Normalization in Machine Learning helps regulating the outputs of neurons, leading to the acceleration of the gradient descent. Normalization improves numerical stability.
Output: Context Precision = 1.00

Example 2:
Question: What is the purpose of Normalization in Machine Learning?
Retrieved Chunks: Normalization in Machine Learning helps regulating the outputs of neurons, leading to the acceleration of the gradient descent. The capital of France is Paris. You can surf the internet using Google.
Output: Context Precision = 0.30

Example 3:
Question: What is the purpose of Normalization in Machine Learning?
Retrieved Chunks: The capital of France is Paris. You can surf the internet using Google. 
Output: Context Precision = 0.00
"""

POST_ANSWER_JUDGE_PROMPT = """
You are a judge. You will evaluate the quality of the input with respect to these metrics.

Faithfulness (Metric)
Faithfulness measures if the answer is supported by the retrieved sources.
A high faithfulness indicates that the answer is supported by the retrieved sources,
while a low faithfulness indicates that the answer is not supported by the retrieved sources. 
If the answer contains claims not present in the retrieved chunks, faithfulness should be low regardless of whether those claims are true.

Answer Relevancy (Metric)
Answer Relevancy measures if the answer is relevant to the question. 
A high answer relevancy indicates that the answer is relevant to the question, 
while a low answer relevancy indicates that the answer is irrelevant to the question.

Answer Correctness (Metric)
Answer Correctness measures if the answer is correct. 
A high answer correctness indicates that the answer is correct, 
while a low answer correctness indicates that the answer is incorrect. 
If there isn't enough information to determine the correctness of the answer, you will give a score of "unavailable".

You will get an input with Question, Retrieved Chunks and Answer section. You have to score the input between 0 and 1. You can use at most 2 decimal numbers while scoring. You will only output the scores like this:
Faithfulness = 0.75
Answer Relevancy = 0.75
Answer Correctness = 0.75

Here are some examples of how to score:
Example:
Question: What is the capital of France?
Retrieved Chunks: Paris is the capital and most populous city of France.
Answer: Paris is the capital of France.
Output:
Faithfulness = 1.00
Answer Relevancy = 1.00
Answer Correctness = 1.00

Example:
Question: What is the capital of France?
Retrieved Chunks: France is a country in Western Europe. It has a rich cultural history.
Answer: The capital of France is Paris.
Output:
Faithfulness = 0.00
Answer Relevancy = 1.00
Answer Correctness = 1.00

Example:
Question: What is the capital of France?
Retrieved Chunks: Paris is the capital of France. Lyon is a major city in eastern France.
Answer: Lyon is located in eastern France.
Output:
Faithfulness = 1.00
Answer Relevancy = 0.10
Answer Correctness = 1.00

Example:
Question: What is the capital of France?
Retrieved Chunks: The capital of France is Lyon.
Answer: The capital of France is Lyon.
Output:
Faithfulness = 1.00
Answer Relevancy = 1.00
Answer Correctness = 0.00

Example:
Question: Who won the men's 100m sprint at the 2024 Summer Olympics?
Retrieved Chunks: The 2024 Summer Olympics were held in Paris. The athletics events took place at Stade de France.
Answer: Noah Lyles won the men's 100m sprint in 2024.
Output:
Faithfulness = 0.00
Answer Relevancy = 1.00
Answer Correctness = 1.00

Example:
Question: Who won the men's 100m sprint at the 2024 Summer Olympics?
Retrieved Chunks: Noah Lyles won the 100m at the 2024 Olympics, clocking 9.79 seconds.
Answer: Kishane Thompson won the men's 100m sprint in 2024.
Output:
Faithfulness = 0.00
Answer Relevancy = 1.00
Answer Correctness = 0.00

Example:
Question: What will be the population of Tokyo in 2030?
Retrieved Chunks: Tokyo is the capital of Japan and one of the most populous cities in the world.
Answer: The population of Tokyo in 2030 will be 37.8 million.
Output:
Faithfulness = 0.00
Answer Relevancy = 1.00
Answer Correctness = unavailable

Example:
Question: What is the capital of France?
Retrieved Chunks: Paris is the capital of France. The Eiffel Tower was built in 1889.
Answer: The capital of France is Paris, and the Eiffel Tower was built in 1889.
Output:
Faithfulness = 1.00
Answer Relevancy = 1.00
Answer Correctness = 1.00

Example:
Question: When did the French Revolution begin?
Retrieved Chunks: Paris is the capital of France. The French Revolution began in 1612.
Answer: The capital of France is Paris and The French Revolution began in 1612.
Output:
Faithfulness = 1.00
Answer Relevancy = 0.60
Answer Correctness = 0.00
"""

FAITHFULNESS_RETRY = "The answer has to be supported by the retrieved chunks. Your previous answer was not supported by the retrieved chunks."
ANSWER_RELEVANCY_RETRY = "The answer has to be relevant to the question. Your previous answer was not relevant to the question."
ANSWER_CORRECTNESS_RETRY = "The answer has to be correct. Your previous answer was not correct."

