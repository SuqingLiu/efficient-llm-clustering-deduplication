import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from collections import defaultdict, Counter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import AzureChatOpenAI
import csv, time, ast

# GPT configuration (Azure OpenAI API)
llm = AzureChatOpenAI(
    openai_api_version="",
    deployment_name="",
    openai_api_key="8",
    openai_api_base="",
    model_version=""
)

# Define the prompt templates for summary and question generation
SUMMARY_PROMPT = """
### System:
Summarize the student-chatbot conversation in a few sentences. The summary should capture the essence of the interaction and key points.

### Conversation:
{conversation}

### Assistant:
"""

QUESTION_PROMPT = """
### System:
Based on the conversation below, generate {num_questions} relevant questions and answers that would be related to the topics discussed in the conversation.

### Conversation:
{conversation}

### Assistant:
"""

QUESTION_TEMPLATE = """
### System:
Below are some comments created by a TA (teaching assistant). Please analyze the comments to see what are the most common issues appearing in these comments. Provide a detailed and specific list of issues identified from these comments. Format the issues as a Python list of dictionaries, where each dictionary contains an 'issue' key and a 'details' key.

### TA:
{new_lines}

### Assistant:
"""

SUMMARY_TEMPLATE = """
### System:
Given the following issue and its related feedbacks, provide a detailed, specific, and actionable summary for students to address this issue. The summary should be helpful and guide the students on how to improve their work based on the feedbacks.

### Issue:
{issue}

### Feedbacks:
{feedbacks}

### Assistant:
"""

# Set up Langchain prompt templates
summary_template = PromptTemplate(input_variables=["conversation"], template=SUMMARY_PROMPT)
question_template = PromptTemplate(input_variables=["conversation", "num_questions"], template=QUESTION_PROMPT)
issue_prompt = PromptTemplate(input_variables=["new_lines"], template=QUESTION_TEMPLATE)
summary_prompt = PromptTemplate(input_variables=["issue", "feedbacks"], template=SUMMARY_TEMPLATE)

summary_chain = LLMChain(llm=llm, prompt=summary_template, output_key="summary")
question_chain = LLMChain(llm=llm, prompt=question_template, output_key="qa")
issue_chain = LLMChain(llm=llm, prompt=issue_prompt, output_key="output")
summary_chain_issue = LLMChain(llm=llm, prompt=summary_prompt, output_key="output")

# Function to preprocess conversation logs and extract them into a single string
def extract_conversation(chatlogs):
    return " ".join([log['chatlog'] for log in chatlogs])

# Function to process conversations in batches and avoid exceeding the token limit
def process_conversations_in_batches(conversations_batch, num_clusters, max_tokens=32768):
    total_tokens = 0
    batch_results = []

    current_batch = []

    for conversation in conversations_batch:
        conversation_text = extract_conversation(conversation['chatlogs'])
        tokens = len(conversation_text.split())

        if total_tokens + tokens > max_tokens:
            # Process the current batch
            result = process_batch_of_conversations(current_batch, num_clusters)
            batch_results.append(result)

            # Reset the batch and token count
            current_batch = []
            total_tokens = 0

        # Add the conversation to the batch
        current_batch.append(conversation)
        total_tokens += tokens

    # Process the final batch if it's not empty
    if current_batch:
        result = process_batch_of_conversations(current_batch, num_clusters)
        batch_results.append(result)

    return batch_results

# Function to process a batch of conversations and generate summary + questions
def process_batch_of_conversations(conversations_batch, num_clusters):
    batch_text = " ".join([extract_conversation(convo['chatlogs']) for convo in conversations_batch])

    # Decide the number of questions based on the number of clusters
    if num_clusters <= 2:
        num_questions = 5
    elif num_clusters <= 5:
        num_questions = 3
    else:
        num_questions = 2

    # Generate summary using GPT for the batch
    summary = summary_chain({"conversation": batch_text})['summary']

    # Generate questions and answers using GPT for the batch
    qa = question_chain({"conversation": batch_text, "num_questions": num_questions})['qa']

    return {
        'summary': summary,
        'qa': qa
    }

# Function to load the conversation data based on 'howmany'
def load_conversations(file_path, howmany='all'):
    with open(file_path, 'r') as file:
        conversations = json.load(file)

    if howmany == 'all':
        return conversations
    else:
        # Ensure 'howmany' is an integer
        howmany = int(howmany)  # Convert howmany to an integer
        return conversations[:howmany]  # Load the first 'howmany' conversations

# Function to cluster conversations using DBSCAN
def cluster_conversations(conversations, eps=1.0, min_samples=2):
    # Extract text data from conversations
    conversation_texts = [extract_conversation(convo['chatlogs']) for convo in conversations]

    # Convert the conversation texts into feature vectors using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(conversation_texts)

    # Perform clustering using DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(X)

    # Organize conversations into clusters
    clusters = defaultdict(list)
    for idx, label in enumerate(dbscan.labels_):
        clusters[label].append(conversations[idx])

    return clusters

# Function to select a representative conversation from each cluster
def select_representative_conversations(clusters):
    representative_conversations = []

    for cluster_id, cluster_conversations in clusters.items():
        if cluster_id == -1:
            continue  # Skip noise

        # Example: Select the longest conversation in the cluster as the representative one
        longest_conversation = max(cluster_conversations, key=lambda convo: len(extract_conversation(convo['chatlogs'])))
        representative_conversations.append(longest_conversation)

    return representative_conversations

# Function to cluster feedbacks
def cluster_feedbacks(feedbacks, eps=1.1, min_samples=3):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(feedbacks)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
    clusters = defaultdict(list)
    for idx, label in enumerate(dbscan.labels_):
        if label != -1:  # -1 indicates noise points that don't belong to any cluster
            clusters[label].append(feedbacks[idx])
    return clusters

# Function to process clusters to identify issues
def process_clusters(clusters):
    issue_counter = Counter()
    issue_map = defaultdict(list)
    processed_comments = set()
    token_limit = 5000  # Adjust token limit based on the language model's maximum token limit

    for cluster_id, feedbacks in clusters.items():
        print(f"Processing cluster {cluster_id} with {len(feedbacks)} feedbacks")
        comments_text = " ".join(feedbacks)

        # Convert the feedbacks into feature vectors using TF-IDF
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform([comments_text])

        # If the cluster exceeds the token limit, split into smaller batches
        if X.shape[1] > token_limit:
            batch_size = token_limit // max(len(vectorizer.transform([f]).data) for f in feedbacks)
            feedback_batches = [feedbacks[i:i + batch_size] for i in range(0, len(feedbacks), batch_size)]
        else:
            feedback_batches = [feedbacks]

        for batch in feedback_batches:
            retry_count = 0
            while retry_count < 3:
                try:
                    issues = issue_chain({"new_lines": str(batch)})['output']
                    tmp = ast.literal_eval(issues)
                    for issue_dict in tmp:
                        issue = issue_dict.get("issue")
                        details = issue_dict.get("details")
                        if issue and details:
                            unique_feedbacks = set(batch) - processed_comments
                            issue_map[issue].extend(unique_feedbacks)  # Map only the new feedbacks to the issue
                            issue_counter[issue] += len(unique_feedbacks)  # Update count with unique feedbacks
                            processed_comments.update(unique_feedbacks)
                    print(f"Processed batch in cluster {cluster_id} with {len(tmp)} issues")
                    break
                except (ValueError, SyntaxError) as e:
                    retry_count += 1
                    print(f'\n#####RESTART######\n', e)
                    time.sleep(1)
            if retry_count == 3:
                print(f"Failed to process batch in cluster {cluster_id} after 3 retries.")

    return issue_counter, issue_map

# Function to deduplicate and merge similar issues
def deduplicate_and_merge_issues(issue_counter, issue_map):
    # Deduplicate and merge similar issues
    merged_issues = Counter()
    merged_issue_map = defaultdict(list)

    for issue, count in issue_counter.items():
        found_similar = False
        for merged_issue in merged_issues:
            if are_similar(issue, merged_issue):
                merged_issues[merged_issue] += count
                merged_issue_map[merged_issue].extend(issue_map[issue])
                found_similar = True
                break
        if not found_similar:
            merged_issues[issue] = count
            merged_issue_map[issue].extend(issue_map[issue])

    return merged_issues, merged_issue_map

# Function to check if two issues are similar
def are_similar(issue1, issue2):
    # Define similarity criteria (this can be more complex if needed)
    issue1_set = set(issue1.lower().split())
    issue2_set = set(issue2.lower().split())
    common_words = issue1_set.intersection(issue2_set)
    return len(common_words) / max(len(issue1_set), len(issue2_set)) > 0.5

# Function to generate summary for an issue
def generate_summary(issue, feedbacks):
    feedbacks_text = "\n".join(feedbacks)
    retry_count = 0
    while retry_count < 3:
        try:
            print(f"Generating summary for issue: {issue}")
            summary = summary_chain_issue({"issue": issue, "feedbacks": feedbacks_text})['output']
            return summary
        except (ValueError, SyntaxError) as e:
            retry_count += 1
            print(f'\n#####RESTART######\n', e)
            time.sleep(1)
    print(f"Failed to summarize issue: {issue} after 3 retries.")
    return ""

# Function to get top issues
def get_top_issues(issue_counter, n=10):
    top_issues = issue_counter.most_common(n)
    return top_issues

# Main execution logic
def main():
    # Helper variable to control how many conversations to load
    howmany = 'all'  # Set '1' to load the first conversation, '2' for two conversations, or 'all' to load all conversations

    # Sample usage to load conversations from a JSON file
    file_path = '/Users/suqingliu/Downloads/conversation_data.json'
    conversation_data = load_conversations(file_path, howmany)

    if len(conversation_data) == 1:
        # If only one conversation is loaded, process it directly without clustering
        print(f"Processing single conversation (no clustering needed)")
        result = process_batch_of_conversations([conversation_data[0]], num_clusters=1)

        # Display the result
        print(f"Summary: {result['summary']}")
        print(f"Questions and Answers:\n{result['qa']}\n")
    else:
        # Cluster the loaded conversations
        clusters = cluster_conversations(conversation_data)

        # Select representative conversations from each cluster
        representative_conversations = select_representative_conversations(clusters)

        # Process the selected key conversations in batches to avoid exceeding token limits
        num_clusters = len([c for c in clusters.keys() if c != -1])  # Count valid clusters
        batch_results = process_conversations_in_batches(representative_conversations, num_clusters, max_tokens=32768)

        # Combine summaries from all batches to generate a generalized theme
        all_summaries = " ".join([batch['summary'] for batch in batch_results])
        general_theme = summary_chain({"conversation": all_summaries})['summary']

        # Display the generalized theme
        print(f"General Theme Across Clusters: {general_theme}\n")

        # Display the results for each batch
        for i, batch in enumerate(batch_results):
            print(f"Batch {i + 1} Summary: {batch['summary']}")
            print(f"Questions and Answers:\n{batch['qa']}\n")

        # Load feedbacks from CSV file
        file_name = "csc413_proposal_onetimeannotations.csv"
        feedbacks = []
        with open(file_name, mode='r', newline='') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                feedbacks.append(row[3])

        # Cluster feedbacks
        clusters = cluster_feedbacks(feedbacks, eps=1.1, min_samples=4)

        # Print the total number of clusters
        print(f"Total number of clusters: {len(clusters)}")

        # Process clusters to identify issues
        issue_counter, issue_map = process_clusters(clusters)

        # Deduplicate and merge similar issues
        merged_issues, merged_issue_map = deduplicate_and_merge_issues(issue_counter, issue_map)

        # Get the top 10 issues
        top_issues = get_top_issues(merged_issues, 10)
        print("Top Issues:", top_issues)

        # Generate summaries for the top issues
        final_results = {
            "top_issues": {issue: count for issue, count in top_issues},
            "issue_map": {issue: merged_issue_map[issue] for issue, count in top_issues},
            "summaries": {}
        }

        for issue, _ in top_issues:
            if issue not in final_results["issue_map"] or not final_results["issue_map"][issue]:
                print(f"No feedbacks found for issue: {issue}")
            summary = generate_summary(issue, final_results["issue_map"].get(issue, []))
            final_results["summaries"][issue] = summary

        # Print the summaries
        for issue, summary in final_results["summaries"].items():
            print(f"Issue: {issue}\nSummary: {summary}\n")

        # Save the results and summaries to the output file
        output_file = "output.json"
        with open(output_file, 'w') as file:
            json.dump(final_results, file, indent=4)

# Run the main function
if __name__ == "__main__":
    main()
