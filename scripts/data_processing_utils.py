import pandas as pd

def create_topic_dataframes(topics):
    if not topics:
        return None, None

    topic_df = pd.DataFrame({
        'Topic_Number': range(1, len(topics) + 1),
        'Topic_Label': [topic[0] for topic in topics],
        'Coherence_Score': [topic[2] for topic in topics]
    })

    words_data = []
    for topic_num, (_, words, _) in enumerate(topics, 1):
        for word in words:
            words_data.append({
                'Topic_Number': topic_num,
                'Word': word
            })

    words_df = pd.DataFrame(words_data)

    return topic_df, words_df