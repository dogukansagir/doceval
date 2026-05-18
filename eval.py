from rag import retrieve
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import config
import prompts

main_llm = ChatOpenAI(model_name=config.DEEPSEEK_FAST_MODEL, api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_API_URL, temperature=0.3)
judge_llm = ChatOpenAI(model_name=config.DEEPSEEK_GOOD_MODEL, api_key=config.DEEPSEEK_API_KEY, base_url=config.DEEPSEEK_API_URL, temperature=0.05)

def parse_output(output):
    lines = output.strip().split("\n")
    scores = {}
    for line in lines:
        if "=" in line:
            metric, value = line.split("=")
            metric = metric.strip()
            value = value.strip()
            try:
                scores[metric] = float(value)
            except ValueError: # in case of "Unavailable"
                scores[metric] = value.capitalize()
    return scores

def compare_scores_with_threshold(scores: dict):
    result_dict = {}
    thresholds = {"Faithfulness": config.FAITHFULNESS_THRESHOLD, "Answer Relevancy": config.ANSWER_RELEVANCY_THRESHOLD, "Answer Correctness": config.ANSWER_CORRECTNESS_THRESHOLD}
    for key, threshold in thresholds.items():
        score = scores[key]
        if score == "Unavailable":
            result_dict[key] = "Unavailable"
        elif score >= threshold:
            result_dict[key] = "PASS"
        else:
            result_dict[key] = "FAIL"

    if "FAIL" in result_dict.values():
        result_dict["Overall"] = "FAIL"
    else:
        result_dict["Overall"] = "PASS"
    
    return result_dict

def evaluate(query, vectorstore, bm25_retriever, verbose = False, chat_history = None):
    if chat_history is None:
        chat_history = []
    cp_retry_count = 0
    answer_retry_count = 0
    # Rewriting the question in order to make it history-agnostic and more suitable for retrieval.
    query_rewriting_prompt = ChatPromptTemplate.from_messages([
    ("system", f"{prompts.QUERY_REWRITE_PROMPT}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "Question: {question}")
    ])

    chain = query_rewriting_prompt | main_llm | StrOutputParser()
    rewritten_query = chain.invoke({"question": query, "chat_history": chat_history}).strip()

    # RETRIEVAL PART STARTS HERE
    retrieved_chunks = retrieve(rewritten_query, vectorstore, bm25_retriever)
    retrieved_texts = [chunk.page_content for chunk in retrieved_chunks]

    judge_context_precision_prompt = ChatPromptTemplate.from_messages([
    ("system", f"{prompts.CONTEXT_PRECISION_PROMPT}"),
    ("human", "Question: {question}\nRetrieved Chunks: {retrieved_chunks}")
    ])
    
    chain = judge_context_precision_prompt | judge_llm | StrOutputParser()

    context_precision_output = chain.invoke({"question": rewritten_query, "retrieved_chunks": "\n".join(retrieved_texts)})
    context_precision_score = parse_output(context_precision_output)["Context Precision"]

    if verbose:
        print(f"Context Precision Score in Retry {cp_retry_count}: {context_precision_score}")

    best_context_precision_score = context_precision_score
    best_retrieved_chunks = retrieved_texts

    while context_precision_score < config.CONTEXT_PRECISION_THRESHOLD and cp_retry_count < config.CONTEXT_PRECISION_RETRY_COUNT:
        # Re-retrieve chunks with different search weights
        cp_retry_count += 1
        retrieved_chunks = retrieve(rewritten_query, vectorstore, bm25_retriever, cosine_weight = min(1.0, config.COSINE_WEIGHT + 0.1*cp_retry_count), bm25_weight = max(0.0, config.BM25_WEIGHT - 0.1*cp_retry_count))
        retrieved_texts = [chunk.page_content for chunk in retrieved_chunks]

        context_precision_output = chain.invoke({"question": rewritten_query, "retrieved_chunks": "\n".join(retrieved_texts)})
        context_precision_score = parse_output(context_precision_output)["Context Precision"]

        if verbose:
            print(f"Context Precision Score in Retry {cp_retry_count}: {context_precision_score}")

        if context_precision_score > best_context_precision_score:
            best_context_precision_score = context_precision_score
            best_retrieved_chunks = retrieved_texts
    
    retrieved_texts = best_retrieved_chunks
    context_precision_score = best_context_precision_score
    # RETRIEVAL PART ENDS HERE

    main_llm_prompt = ChatPromptTemplate.from_messages([
    ("system", f"{prompts.MAIN_LLM_PROMPT}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "Question: {question}\nRetrieved Chunks: {retrieved_chunks}")
    ])

    chain = main_llm_prompt | main_llm | StrOutputParser()
    answer = chain.invoke({"question": query, "retrieved_chunks": "\n".join(retrieved_texts), "chat_history": chat_history})

    judge_post_answer_prompt = ChatPromptTemplate.from_messages([
    ("system", f"{prompts.POST_ANSWER_JUDGE_PROMPT}"),
    ("human", "Question: {question}\nRetrieved Chunks: {retrieved_chunks}\nAnswer: {answer}")
    ])

    chain = judge_post_answer_prompt | judge_llm | StrOutputParser()
    post_answer_judge_output = chain.invoke({"question": rewritten_query, "retrieved_chunks": "\n".join(retrieved_texts), "answer": answer})
    post_answer_scores = parse_output(post_answer_judge_output)
    
    if verbose:
        print(f"Attempt {answer_retry_count} scores: {post_answer_scores}")

    best_metric_scores = post_answer_scores
    best_answer = answer

    status = compare_scores_with_threshold(best_metric_scores)

    while status["Overall"] == "FAIL" and answer_retry_count < config.RETRY_COUNT:
        # Generate a new answer with the same retrieved chunks with main_llm prompt that includes feedback based on which metric failed.
        answer_retry_count += 1

        to_be_added_prompts = []
        if status["Faithfulness"] == "FAIL":
            to_be_added_prompts.append(prompts.FAITHFULNESS_RETRY)

        if status["Answer Relevancy"] == "FAIL":
            to_be_added_prompts.append(prompts.ANSWER_RELEVANCY_RETRY)

        if status["Answer Correctness"] == "FAIL":
            to_be_added_prompts.append(prompts.ANSWER_CORRECTNESS_RETRY)

        main_llm_prompt = ChatPromptTemplate.from_messages([
        ("system", f"{prompts.MAIN_LLM_PROMPT + ' '.join(to_be_added_prompts)}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "Question: {question}\nRetrieved Chunks: {retrieved_chunks}")
        ])
        
        chain = main_llm_prompt | main_llm | StrOutputParser()
        answer = chain.invoke({"question": query, "retrieved_chunks": "\n".join(retrieved_texts), "chat_history": chat_history})

        judge_post_answer_prompt = ChatPromptTemplate.from_messages([
        ("system", f"{prompts.POST_ANSWER_JUDGE_PROMPT}"),
        ("human", "Question: {question}\nRetrieved Chunks: {retrieved_chunks}\nAnswer: {answer}")
        ])

        chain = judge_post_answer_prompt | judge_llm | StrOutputParser()
        post_answer_judge_output = chain.invoke({"question": rewritten_query, "retrieved_chunks": "\n".join(retrieved_texts), "answer": answer})
        post_answer_scores = parse_output(post_answer_judge_output)

        if verbose:
            print(f"Attempt {answer_retry_count} scores: {post_answer_scores}")

        if post_answer_scores["Answer Correctness"] != "Unavailable" and best_metric_scores["Answer Correctness"] != "Unavailable":
            if sum(post_answer_scores.values()) > sum(best_metric_scores.values()):
                best_metric_scores = post_answer_scores
                best_answer = answer
        else:
            if post_answer_scores["Faithfulness"] + post_answer_scores["Answer Relevancy"] > best_metric_scores["Faithfulness"] + best_metric_scores["Answer Relevancy"]:
                best_metric_scores = post_answer_scores
                best_answer = answer

        status = compare_scores_with_threshold(post_answer_scores)

    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=best_answer))
    return best_answer, best_metric_scores, context_precision_score, chat_history