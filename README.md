# Enhancing Search Engine Relevance for Video Subtitles (Cloning Shazam)

## Background:
In the fast-evolving landscape of digital content, effective search engines play a pivotal role in connecting users with relevant information. For platforms like Google, providing a seamless and accurate search experience is paramount. This project focuses on improving the search relevance for video subtitles, enhancing the accessibility of video content.

## Objective:
Develop an advanced search engine algorithm that efficiently retrieves subtitles based on user queries, with a specific emphasis on subtitle content. The primary goal is to leverage natural language processing (NLP) and machine learning techniques to enhance the relevance and accuracy of search results.

## Keyword-based vs. Semantic Search Engines:

### Keyword-Based Search Engine:
- These search engines rely heavily on exact keyword matches between the user query and the indexed documents.
- Suitable for retrieving documents with precise keyword occurrences but lacks understanding of contextual meaning.

### Semantic Search Engine:
- Goes beyond simple keyword matching to understand the meaning and context of user queries and documents.
- Uses embeddings and similarity metrics to provide more relevant and meaningful search results.

### Comparison:
- While keyword-based search engines focus primarily on matching exact words in documents, semantic search engines aim to understand the deeper meaning and context of user queries, resulting in more accurate and intelligent retrieval.

## Core Logic:
To compare a user query against a video subtitle document, the core logic involves three key steps:

### 1. Preprocessing of Data:
- If compute resources are limited, a random 10% to 30% of the data can be sampled.
- **Cleaning**: Remove timestamps and other non-textual metadata.
- **Vectorization**: Convert cleaned subtitle text into numerical representations.
- **Query Vectorization**: Encode user query into a similar vector representation.

### 2. Cosine Similarity Calculation:
- Compute cosine similarity between the vector of the subtitle documents and the vector of the user query.
- This similarity score determines the relevance of the documents to the user's query.
- Return the top-matching subtitle documents.

## Data:
- The dataset consists of video subtitles stored in a `.db` file format.
- Download and extract subtitle data.
- Understand database structure and decode relevant subtitle information.

## Step-by-Step Process:

### Part 1: Ingesting Documents
1. Read and load the subtitle data.
2. Analyze the `.db` file and extract relevant subtitle content.
3. Decode and clean subtitle text by removing timestamps and unwanted characters.
4. If necessary, take a random sample (30%) of the data to manage computational resources.
5. Experiment with different text vectorization techniques:
   - **BOW / TF-IDF**: Generates sparse vector representations (suitable for keyword-based search engines).
   - **BERT-based SentenceTransformers**: Generates dense embeddings that encode semantic information (suitable for semantic search engines).

### (Must Implement) Document Chunking:
- **Problem**: Large subtitle documents may result in information loss when embedding the entire document as a single vector.
- **Solution**: Divide large documents into smaller, manageable chunks before embedding.
- **Overlapping Window Strategy**: To avoid cutting off important context, create overlapping chunks with shared tokens.
- **Storage**: Store embeddings efficiently in a **ChromaDB** database.

### Part 2: Retrieving Documents
1. Accept user search query in **audio format**.
2. **Transcribe** audio into text using **AssemblyAI**.
3. Preprocess the transcribed text (cleaning, formatting, etc.).
4. Generate **query embedding** using the same method used for subtitle embeddings.
5. Compute **cosine similarity** between query embedding and subtitle embeddings stored in **ChromaDB**.
6. Retrieve the most relevant subtitle documents based on similarity scores.
7. Display results, allowing users to adjust the number of returned results dynamically.

## Implementation Stack:
- **Data Processing**: Python (pandas, numpy)
- **Vectorization**: SentenceTransformers, TF-IDF
- **Embedding Storage**: ChromaDB
- **Audio Transcription**: AssemblyAI
- **Web Application**: Streamlit

By implementing this step-by-step approach, the project will improve the relevance of subtitle search results, making video content more accessible and searchable.

