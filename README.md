 # 💬 RAG Intelligence AI

A simple and powerful Chatbot that can "read" your documents and answer questions based on them. Instead of just talking to a general AI, you can give this AI your own PDFs, URLs, or YouTube videos to make it an expert on your specific data.

Basically its a Multi-Source Chatbot using Langchain

Whats's RAG?
This is a **(Retrieval-Augmented Generation)** application. 
- **Retrieval:** It searches through the files you upload to find the most relevant information.
- **Augmented:** It adds that information to your question.
- **Generation:** It uses a powerful Brain (LLM) to write a clear answer based only on your data.

# FLOW #
1. Loaders: First, the app "reads" your data (PDFs, URLs, YouTube transcripts). It turns different file types into a standard text format the computer can understand.

2. RecursiveTextSplitter: AI cannot read a 100-page book all at once. I used this to "chop" your text into small, 500-character chunks with a little bit of "overlap" (100 characters).
Why? Overlapping ensures that a sentence cut in half still makes sense in the next chunk. It helps the AI keep the context of the story.

3. Embeddings: These chunks are turned into "number maps" (Vectors).

4. Vectorstore (Chroma): All those "number maps" are stored in a digital library called Chroma.

5. Retrieval & Search Queries: When you ask a question, the app doesn't search for exact words; it searches for meaning. It pulls the top 4 chunks from the library that are most similar to your question.

6. LLM (The Brain): Finally, we send those 4 chunks + your question to the "llama-3.3-70b-versatile" model.

7. Generate Output: The LLM reads the snippets and writes a natural answer for you.


I used three specific LangChain tools to make the conversation feel "human":
1. create_history_aware_retriever: Most AIs forget what you just said. If you ask "Who is Steve Jobs?" and then "When was he born?", a normal AI might not know who "he" is. This retriever "rewrites" your second question to include the context of the first one.

2. create_stuff_documents_chain: This is the "Stuffer." It takes all those text chunks we found in the library and "stuffs" them into the prompt for the LLM to read.

3. create_retrieval_chain: This is the final link. It connects the Retriever (the searcher) with the Stuff Chain (the writer) to give you a complete answer.


Extracting the sources:
I didn't want the AI to just "claim" things. I programmed the app to look into the "metadata" of the chunks it found. This way, the app can show you a "View Sources" button, proving exactly which PDF page or Website URL it used to find the answer.

Chat Sessions (Memory):
Using st.session_state, the app remembers your chat history during your current visit. It stores the list of messages so you can see your full conversation on the screen, just like WhatsApp or ChatGPT.