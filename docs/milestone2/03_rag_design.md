# Milestone 2 - RAG design

## What RAG is used for

- We use two separate Chroma collections, because the two jobs need different data and a different kind of search. In one shared collection, adoption listings would compete with knowledge chunks for the same top results.
- Knowledge collection: breed and species profiles. Used to answer questions about facts, danger, care needs and lifestyle fit.
- Listings collection: real shelter adoption listings (name, species, breed, colour, size, description). Used to suggest names: we find pets that look and act like the user's pet and use their names.
- Name suggestion covers dogs and cats only, because the listing data only contains dogs and cats.

## Data and new data

- Knowledge sources: Wikipedia breed and species articles, TheDogAPI and TheCatAPI breed data.
- Listing sources: PetFinder.my Adoption Prediction (Kaggle, about 15,000 dog and cat listings). The final choice is made in the Problem and data section.
- Cleaning: remove duplicate documents, listings with placeholder names ("No Name", "Puppy 1") and listings for a whole litter.
- New data: add the new files to the data folder and re-run the ingestion script.
- Every chunk has a stable ID, so Chroma upsert updates existing chunks and adds new ones without creating duplicates.

## Chunking

- Knowledge: one chunk per section of a profile, for example "Siberian Husky - Exercise". Most questions ask about one topic of one breed (exercise, food, danger), so one section usually holds the whole answer.
- Sections longer than about 250 words are split again, with a small overlap, so each chunk stays focused and does not fill the prompt with unrelated text.
- Each knowledge chunk starts with the breed and section name, so a chunk such as "needs two hours of exercise a day" still says which breed it is about.
- Baseline to compare against: fixed windows of 90 words with 15 words of overlap (the Lab 3 setting). It needs no structure in the documents, so it shows whether splitting by section is worth the extra work.
- Listings: one listing = one chunk, never split, because listings are short and their fields only make sense together. The text is species, breed, colour, size, age and the description.
- The pet's name is kept in metadata only, not in the embedded text, because the search query never contains a name.

## Embedding model

- Main model: nomic-embed-text-v1.5, run locally in LM Studio.
- Why: it is the model used in the course labs, it runs on our laptops at no cost and it accepts long text (up to 8,192 tokens).
- We use the prefixes the model expects: "search_document: " for chunks and "search_query: " for questions, because the model was trained with them and its documentation requires them.
- Compared with BM25 keyword search: no neural model, the baseline from the lab. Breed names such as "Shiba Inu" are exact words, where keyword search can beat embeddings, so we measure it instead of assuming.
- Compared with bge-small-en-v1.5: a smaller, faster model, to check whether the bigger one is worth it.
- One collection per embedding model; vectors from different models are never mixed, because they are not comparable.

## Vector database

- Chroma, saved locally on disk, with cosine distance.
- Why: we used it in Lab 3, it needs no server, it stores text, vector and metadata together, and it can filter by metadata (species, breed, colour).
- Our data is small (a few thousand knowledge chunks and about 15,000 listings), so a database server is not needed.
- Not chosen: FAISS (no metadata filtering or storage out of the box), Qdrant and pgvector (both need a separate server).

## RAG approaches

- Basic RAG (baseline): embed the question, take the top 3 chunks, and the SLM answers only from them, with citations. It is the simplest version, so it shows what the other approaches add.
- Corrective RAG: the SLM first checks whether the chunks really answer the question. If not, it rewrites the query and searches one more time. Used for danger and care questions, where a made-up answer could harm the pet or the user.
- Agentic RAG: the agent decides which collection to search and with which filters, with at most 2 searches per question. Needed for mixed questions such as "Would this dog suit my small flat, and what should I name it?"
- HyDE: considered but not chosen. Our questions are short and clear, and an invented passage could pull the search toward the wrong facts on safety questions.
- When the breed is known (from the photo or the chat), we filter by breed in Chroma instead of adding keyword search on top of the vector search. An exact filter is simpler and more reliable than combining two search methods.
- Name suggestion: filter listings by species and main colour, rank them by similarity to the description of the user's pet (from the photo and the chat), and take the 20 closest.
- Species and colour are separate fields in the listings, so an exact filter is more reliable than hoping the vectors capture colour. Similarity then ranks by everything else: size, breed and personality.
- The most common names among those 20 pets go to the SLM, which picks 5 and explains each choice. A name that appears often among similar pets is a name people really chose for animals like this one, and the explanation lets the user see why it was suggested.

## Rules for answers

- The SLM answers only from the retrieved chunks and cites their IDs.
- Code checks that every cited ID was really retrieved.
- If the chunks do not answer the question, the assistant says it does not know. It never says an animal or a food is safe without a source.

## How we compare the options

- Retrieval: hit rate at 1 and at 3 on the golden questions (is the right source among the top results).
- Answers: correct facts, valid citations, and a correct "I don't know" on questions the data cannot answer.
- Cost: number of SLM calls and response time for each approach.
- We change one thing at a time (chunking, embedding model or RAG approach) and keep the same questions.
