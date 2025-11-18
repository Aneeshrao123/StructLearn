def generate_answer(question: str, ranked_context):
    """
    Simple answer generator:
    Takes the top-ranked RAG resource and turns it into a final explanation.
    """
    if not ranked_context:
        return "Sorry, I could not find relevant material."

    # Take the highest similarity match
    best_topic, explanation, score = ranked_context[0]

    answer = (
        f"### Explanation for: {best_topic}\n"
        f"{explanation}\n\n"
        f"(Relevance score: {round(score, 2)})"
    )
    
    return answer
