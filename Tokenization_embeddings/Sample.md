# Understanding Tokenization and Embeddings in LLMs

November 13, 2025 by Peter Song

Large language models have transformed how we interact with AI, but their impressive capabilities rest on two fundamental processes that most users never see: tokenization and embeddings. Understanding tokenization and embeddings in LLMs is essential for anyone working with these systems, whether you're optimizing API costs, debugging unexpected behavior, or building applications that leverage language models effectively.

These processes form the bridge between human language and mathematical operations that neural networks can perform. Text doesn't naturally exist in a format that machine learning models can process—tokenization breaks language into manageable pieces, while embeddings convert those pieces into numerical representations that capture semantic meaning. The choices made in these processes profoundly impact model performance, cost, and behavior in ways that aren't immediately obvious.

## What Tokenization Really Means in Practice

Tokenization is the process of breaking text into discrete units called tokens that serve as the basic building blocks for language model processing. Unlike simple word splitting, modern tokenization employs sophisticated algorithms that balance vocabulary size, computational efficiency, and semantic coherence. The most common approach in contemporary LLMs uses subword tokenization methods like Byte Pair Encoding (BPE) or WordPiece.

Subword tokenization operates on a crucial insight: while languages contain hundreds of thousands of unique words, many share common components. Instead of treating every possible word as a unique token, subword methods break text into frequently occurring subunits. The word "unhappiness" might become three tokens: "un", "happi", and "ness". This approach provides several advantages that make modern LLMs practical.

### The Vocabulary Size Tradeoff

The Vocabulary Size Tradeoff represents one of the most important decisions in tokenization design. A smaller vocabulary (around 30,000 tokens) means each word splits into more pieces, requiring more tokens per sentence but less memory to store the vocabulary. A larger vocabulary (100,000+ tokens) keeps more words intact as single tokens, reducing sequence length but increasing the model's memory footprint. Most leading models like GPT-4 use vocabularies of 50,000-100,000 tokens, striking a balance between these competing concerns.

Consider how different tokenization schemes handle the same text. The sentence "The researcher's breakthrough revolutionized AI development" might be tokenized as:

- Character-level tokenization (inefficient): ["T", "h", "e", " ", "r", "e", "s", "e", "a", "r", "c", "h", "e", "r", "'", "s", ...] - 56 tokens
- Word-level tokenization (inflexible): ["The", "researcher", "'", "s", "breakthrough", "revolutionized", "AI", "development"] - 8 tokens, but "researcher's" causes problems
- Subword tokenization (optimal): ["The", " researcher", "'s", " break", "through", " revolution", "ized", " AI", " development"] - 9 tokens with flexible handling of compound words

The tokenization scheme directly affects costs when using LLM APIs. Since providers charge per token, understanding how your text tokenizes helps predict expenses. Common English words typically map to single tokens, but technical terms, non-English languages, special characters, and code often fragment into multiple tokens. A 1,000-word English essay might consume 1,300-1,500 tokens, while equivalent Chinese text could require 2,000-3,000 tokens due to how character-based languages tokenize.

## How Tokenization Impacts Model Behavior

The seemingly technical decision of how to tokenize text creates surprising behavioral consequences that developers must understand. Tokenization artifacts explain many puzzling LLM behaviors that appear to be model limitations but actually stem from how text was split during preprocessing.

### Spelling and Character-Level Tasks

Spelling and Character-Level Tasks suffer when words split across multiple tokens. If you ask an LLM to count letters in "strawberry", the model sees ["str", "awber", "ry"] rather than individual characters. The model must reason about token boundaries and reconstruct character sequences, making simple counting surprisingly difficult. This limitation isn't about the model lacking counting ability—it's about tokenization obscuring the information needed for the task.

Models struggle with tasks like reversing strings, identifying palindromes, or counting specific characters because these operations require character-level awareness that tokenization deliberately abstracts away. When you encounter these limitations, you're seeing tokenization working as designed—optimizing for natural language understanding at the expense of character-level precision.

### Cross-Lingual Performance

Cross-Lingual Performance varies dramatically based on tokenization decisions. English-centric tokenizers trained primarily on English text represent English words efficiently but fragment other languages excessively. The phrase "Hello, how are you?" might be 6 tokens, while the Thai equivalent "สวัสดี คุณเป็นอย่างไร" could require 20+ tokens because each character or small character cluster becomes a separate token.

This tokenization inefficiency compounds costs and performance issues. More tokens mean higher API costs, longer sequences that consume more context window space, more computational work for the model, and potentially degraded understanding as semantic units fracture. Newer multilingual models address this through multilingual tokenizer training, but legacy systems still exhibit strong language biases rooted in tokenization.

### Special Characters and Code

Special Characters and Code present particular challenges. Programming languages use symbols like {, }, =>, != that may tokenize as separate units or combine unpredictably. The code snippet x != y might become ["x", " !=", " y"] or ["x", " !", "=", " y"] depending on the tokenizer's training. These variations affect how well models understand code structure and generate syntactically correct programs.

Mathematical expressions face similar issues. The equation 3.14159 * r^2 tokenizes differently than 3.14 * r^2 because the longer decimal splits into more tokens. Models may show subtle performance differences on mathematically equivalent expressions due purely to tokenization artifacts, not mathematical reasoning capacity.

## The Transformation into Embedding Space

Once text becomes tokens, embeddings transform these discrete symbols into continuous vector representations that neural networks can process. An embedding is a dense vector of floating-point numbers (typically 768, 1024, or higher dimensions) that represents a token's meaning in geometric space. This transformation from symbols to numbers enables mathematical operations that capture semantic relationships.

Think of embeddings as coordinates in meaning-space. Just as GPS coordinates specify locations on Earth's surface, embeddings specify locations in a high-dimensional space where semantic similarity corresponds to geometric proximity. Tokens with related meanings cluster together in this space, while unrelated concepts drift apart. The embedding for "king" sits near "queen", "monarch", and "royalty", while far from "bicycle", "electron", or "jazz".

### The Embedding Matrix

The Embedding Matrix forms one of the largest components of language model architecture. For a model with 50,000 tokens and 1024-dimensional embeddings, the embedding matrix contains 51.2 million parameters just for token representation. Each token maps to a learned vector that captures both the token's inherent meaning and how it typically interacts with other tokens in context.

These embeddings aren't handcrafted—they emerge through training. Initially randomized vectors gradually shift based on the model's task (typically predicting next tokens). Tokens that appear in similar contexts develop similar embeddings because the model learns they serve interchangeable roles. The embedding for "cat" resembles "dog" more than "car" because "cat" and "dog" appear in comparable contexts: both can "run", "sleep", "eat", and "play".

## Contextual Embeddings and the Role of Attention

Static embeddings represent only the beginning of how LLMs process language. Modern transformer-based models use contextual embeddings that change based on surrounding words, enabling the same token to have different meanings in different contexts. This context-sensitivity explains why contemporary LLMs vastly outperform earlier word embedding approaches like Word2Vec or GloVe.

Consider the word "bank" in these sentences:

- "I deposited money at the bank" (financial institution)
- "We picnicked on the river bank" (riverside)
- "The plane performed a steep bank" (turning maneuver)

A static embedding would assign "bank" identical representation in all contexts. Contextual embeddings adapt the representation based on nearby words, allowing the model to distinguish financial, geographical, and aeronautical meanings through attention mechanisms.

### Attention Mechanisms

Attention Mechanisms enable this contextual adaptation by letting each token "look at" other tokens in the sequence and adjust its representation accordingly. When processing "The cat sat on the mat", the embedding for "sat" attends to "cat" (the subject performing the action) and "mat" (the location), modifying its representation to capture this specific instance of sitting.

The attention process involves three learned transformations for each token: queries (what information this token seeks), keys (what information this token offers), and values (the actual information to incorporate). Tokens with matching queries and keys exchange information, updating their representations to reflect context. This happens across multiple attention heads and layers, building increasingly sophisticated contextual representations.

### Layer-by-Layer Refinement

Layer-by-Layer Refinement progressively enhances embeddings as they pass through the model's layers. Early layers capture surface-level patterns like syntax and word relationships. Middle layers develop semantic understanding and thematic connections. Deep layers encode abstract reasoning, task-specific knowledge, and complex inference capabilities. By the final layer, each token's embedding reflects not just the word itself but its role in the sentence, paragraph, and broader discourse.

Research has shown that different layers encode different types of information. When you extract embeddings from an LLM for downstream tasks, choosing the right layer matters. Early-layer embeddings work well for syntax-sensitive tasks, middle-layer embeddings excel at semantic similarity, and late-layer embeddings capture task-specific representations if the model was fine-tuned for particular applications.

## Practical Applications of Embeddings

Understanding embeddings unlocks numerous practical applications beyond text generation. Modern NLP workflows leverage embeddings for semantic search, clustering, classification, and recommendation systems—often using specialized embedding models trained specifically for these tasks.

### Semantic Search

Semantic Search relies on embeddings to find relevant documents based on meaning rather than keyword matching. Traditional search finds documents containing specific words; semantic search finds documents with related concepts. Query "affordable transportation options" matches articles about "budget-friendly vehicles" and "economical travel methods" even without word overlap.

Implementing semantic search involves embedding both your document corpus and user queries, then finding documents whose embeddings are geometrically closest to the query embedding. Cosine similarity typically measures this closeness, yielding values from -1 (opposite meaning) to 1 (identical meaning). Documents with similarity above a threshold (often 0.7-0.8) return as relevant results.

### Clustering and Classification

Clustering and Classification leverage embeddings to group similar content or categorize new items. Customer support tickets embedded and clustered automatically group by topic—billing issues cluster together, technical problems form another group, feature requests a third. This automated organization helps route tickets, identify trending issues, and allocate resources effectively.

For classification tasks, you can train simple models (like logistic regression or small neural networks) on embeddings rather than raw text. The embeddings capture semantic features, allowing classifiers to generalize better with less training data. A sentiment classifier trained on embeddings often outperforms one trained on word counts or TF-IDF vectors.

### Retrieval-Augmented Generation

Retrieval-Augmented Generation combines embeddings with LLM generation for applications requiring factual accuracy. When a user asks a question, you embed the query, retrieve relevant documents using semantic search, and include retrieved content in the LLM's context. The model generates responses grounded in retrieved information rather than relying solely on training data.

This approach mitigates hallucination by anchoring responses to retrieved facts. Instead of generating potentially false information from its parameters, the model cites or paraphrases retrieved content. Embedding quality directly impacts retrieval accuracy—poor embeddings return irrelevant documents, leading to incorrect or nonsensical responses.

## The Relationship Between Tokenization and Embeddings

Tokenization and embeddings form an inseparable pipeline where decisions in one phase affect outcomes in the next. The granularity of tokens determines what semantic units embeddings must capture, while embedding dimensionality influences how much information each token can encode.

Fine-grained tokenization (many small tokens) requires embeddings to capture meaning across token sequences. When "unhappiness" splits into "un", "happi", "ness", embeddings for these fragments must compose into the full word's meaning. The model learns compositional semantics—how "un" modifies "happi" to reverse sentiment—through training on countless examples. This compositionality enables models to handle novel compound words not seen during training.

Coarse-grained tokenization (fewer, larger tokens) embeds more complete semantic units directly but struggles with vocabulary coverage. Rare words or typos become out-of-vocabulary tokens, typically mapped to a special [UNK] token with a generic embedding that provides little semantic information. Modern subword tokenization balances these extremes, ensuring virtually all input can be represented through token combinations while maintaining reasonable vocabulary size.

The training process jointly optimizes tokenization and embeddings. During model training, the embedding matrix updates to better predict next tokens given context. Frequently co-occurring token pairs develop embeddings that work well together, while rarely seen combinations remain poorly optimized. This explains why LLMs excel at common language patterns but struggle with unusual formulations—the embedding space reflects training data statistics.

## Optimizing Tokenization and Embedding Usage

Developers working with LLMs can optimize performance and costs through strategic tokenization and embedding management. Understanding these mechanisms enables more efficient prompting, better model selection, and improved application design.

### Prompt Engineering for Token Efficiency

Prompt Engineering for Token Efficiency reduces costs and fits more information in context windows. Instead of verbose instructions, use concise phrasing that achieves the same semantic effect with fewer tokens. "Please provide a comprehensive and detailed explanation" (8 tokens) conveys essentially the same meaning as "Explain thoroughly" (3 tokens), saving 5 tokens per instruction. At scale, these savings compound significantly.

Structure prompts to maximize information density per token. Technical jargon often encodes more meaning per token than everyday language. "Implement OAuth2 authentication" conveys specific technical requirements in 4 tokens, while a natural language equivalent might require dozens of tokens to specify the same information.

### Model Selection Based on Tokenization

Model Selection Based on Tokenization matters when working with multilingual content or code-heavy applications. Some models use more efficient tokenization for specific domains. Code-specialized models like Codex employ tokenizers trained on programming languages, representing code constructs more efficiently than general-purpose models. Similarly, multilingual models with language-balanced tokenizers reduce costs when processing non-English text.

### Embedding Model Choice

Embedding Model Choice affects downstream application quality. General-purpose embeddings from instruction-tuned LLMs work adequately for many tasks, but specialized embedding models often outperform for specific applications. Models like text-embedding-ada-002 or sentence-transformers are optimized for semantic similarity tasks, producing embeddings that cluster and compare more reliably than general LLM embeddings.

Consider embedding dimensionality tradeoffs. Higher-dimensional embeddings (1536+) capture more nuanced semantic information but require more storage and slower similarity computations. Lower-dimensional embeddings (384-768) sacrifice some semantic fidelity for efficiency. For massive-scale applications, this tradeoff significantly impacts infrastructure costs.

### Caching and Reusing Embeddings

Caching and Reusing Embeddings prevents redundant computation. Once you've embedded a document, store that embedding rather than regenerating it for each query. Build vector databases that persist embeddings with efficient similarity search capabilities. Systems like Pinecone, Weaviate, or Milvus specialize in storing and querying embedding vectors at scale.