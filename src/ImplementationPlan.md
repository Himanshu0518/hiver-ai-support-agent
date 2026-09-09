

The key idea is:

Use the tweet-level CSV as the evidence layer, transform conversations into high-quality historical support cases, build a case-level RAG index, classify the new message, retrieve historically similar resolutions, generate a grounded reply, and finally make an explicit auto-handle/escalate decision.

1. Overall architecture
                    ┌─────────────────────────┐
                    │ AmazonHelp Twitter Data │
                    └────────────┬────────────┘
                                 │
              ┌──────────────────┴──────────────────┐
              │                                     │
              ▼                                     ▼
 amazonhelp_tweets.csv                 amazonhelp_conversations.csv
   (tweet-level evidence)                (conversation-level)
              │                                     │
              └──────────────────┬──────────────────┘
                                 ▼
                    ┌────────────────────────┐
                    │ Data Cleaning + PII     │
                    │ Sanitization            │
                    └────────────┬───────────┘
                                 ▼
                    ┌────────────────────────┐
                    │ Conversation Quality   │
                    │ Filtering               │
                    └────────────┬───────────┘
                                 ▼
                    ┌────────────────────────┐
                    │ Intent Taxonomy         │
                    │ + Intent Labels         │
                    └────────────┬───────────┘
                                 ▼
                    ┌────────────────────────┐
                    │ Historical Resolution  │
                    │ Cases                   │
                    └────────────┬───────────┘
                                 │
                       ┌─────────┴─────────┐
                       ▼                   ▼
                Intent Classifier       RAG Index
                       │                   │
                       └─────────┬─────────┘
                                 ▼
                        Incoming Customer
                              Message
                                 │
                    ┌────────────┴────────────┐
                    ▼                         ▼
              Intent Prediction         Case Retrieval
                    │                         │
                    └────────────┬────────────┘
                                 ▼
                         Evidence Reranking
                                 │
                                 ▼
                         Response Generator
                                 │
                                 ▼
                       Safety / Risk Check
                                 │
                       ┌─────────┴─────────┐
                       ▼                   ▼
                  AUTO-HANDLE          ESCALATE
2. Phase 0 — Repository structure

Keep the repo simple.

hiver-ai-support-agent/
│
├── data/
│   ├── raw/
│   │   └── amazonhelp_tweets.csv
│   │
│   ├── processed/
│   │   └── amazonhelp_conversations.csv
│   │
│   ├── kb/
│   │   └── amazonhelp_cases.csv
│   │
│   └── evaluation/
│       └── golden_set.csv
│
├── src/
│   ├── preprocessing/
│   │   ├── conversations.py
│   │   ├── pii.py
│   │   └── quality.py
│   │
│   ├── intent/
│   │   ├── taxonomy.py
│   │   ├── baseline.py
│   │   └── classifier.py
│   │
│   ├── retrieval/
│   │   ├── index.py
│   │   ├── retrieve.py
│   │   └── reranker.py
│   │
│   ├── generation/
│   │   └── response_generator.py
│   │
│   ├── routing/
│   │   └── router.py
│   │
│   ├── evaluation/
│   │   ├── evaluate_intent.py
│   │   ├── evaluate_retrieval.py
│   │   ├── evaluate_replies.py
│   │   └── evaluate_routing.py
│   │
│   └── pipeline.py
│
├── notebooks/
│   ├── 01_data_analysis.ipynb
│   ├── 02_intent_taxonomy.ipynb
│   ├── 03_build_kb.ipynb
│   └── 04_evaluation.ipynb
│
├── artifacts/
│   ├── intent_model/
│   └── vector_index/
│
├── README.md
├── requirements.txt
└── report.pdf

Don't add LangGraph/CrewAI unless you discover a real need. The assignment is evaluating results and reasoning, not architecture complexity.

3. Phase 1 — Understand the dataset

You already have:

amazonhelp_tweets.csv

Individual tweets:

tweet_id
author_id
inbound
created_at
text
response_tweet_id
in_response_to_tweet_id
conversation_id

This is your ground-truth evidence.

amazonhelp_conversations.csv

Conversation-level representation:

conversation_id
turn_count
customer_problem
conversation
amazon_responses
customer_turns
first_timestamp
last_timestamp

This becomes the starting point for case construction.

4. Phase 2 — Data quality analysis

Before building anything, run an analysis.

Calculate:

Total tweets
Total AmazonHelp-related tweets
Total conversations
Average conversation length
Median conversation length
Customer/agent turn ratio
Percentage 2-turn conversations
Percentage multi-turn conversations
Missing text
Duplicate tweets
Missing parent IDs
Missing response IDs

Also inspect the distribution.

For example:

cases["turn_count"].describe()

And:

cases["turn_count"].value_counts().sort_index()

This becomes useful evidence in your report.

5. Phase 3 — PII sanitization

This is important because support conversations can contain:

order numbers
email addresses
phone numbers
URLs
account identifiers
tracking information

Before putting examples into a public GitHub repository or evaluation artifact, sanitize them.

Example:

Original:

"My order 404-1234567-1234567 hasn't arrived."

↓

"My order [ORDER_ID] hasn't arrived."

Implement:

def sanitize_text(text):
    ...

Patterns for:

EMAIL
PHONE
ORDER_ID
URL
TRACKING_ID

Keep the original tweet CSV private if necessary.

Your public processed dataset should contain sanitized text.

6. Phase 4 — Conversation quality filtering

Not every conversation should become a RAG case.

Create quality rules.

For example:

Keep
Customer has a recognizable problem
+
AmazonHelp provides some useful handling
Lower quality
Customer says "thanks"
AmazonHelp says "you're welcome"
Remove
Spam
Pure compliments
Pure criticism with no support interaction
Empty/garbled text
Unusable conversations

But don't remove too aggressively.

A short:

Customer: My package is late.
AmazonHelp: Please contact support through this link.

can still be a valid historical handling example.

7. Phase 5 — Create intent taxonomy

This is one of the most important parts.

The assignment says the intents should be defined from the data.

So don't start with arbitrary intents from Banking77.

First perform exploratory analysis on AmazonHelp conversations.

Look at:

customer_problem

and cluster/group recurring problems.

You may discover categories such as:

late_delivery
order_cancellation
wrong_item
missing_item
refund_issue
payment_issue
account_issue
gift_card
promotion
product_availability
amazon_pay
delivery_problem
technical_issue
general_support

But don't blindly use this exact list.

The final taxonomy should come from your actual dataset.

8. Keep the taxonomy small

I'd target approximately:

8–15 intents

Not 50.

Why?

Because you need enough examples per intent to:

train
evaluate
retrieve
explain failures

For every intent create a definition:

{
  "intent": "late_delivery",
  "definition": "Customer reports that an expected package has not arrived or is delayed.",
  "examples": [
    "...",
    "..."
  ]
}

Also define confusing neighboring intents.

Example:

late_delivery
vs
missing_package
vs
order_cancellation

This becomes very useful during failure analysis.

9. Phase 6 — Build labelled intent dataset

You don't need to manually label thousands of conversations.

Use a practical approach:

Step 1

Sample conversations.

Step 2

Manually label them.

For example:

conversation_id
customer_problem
intent

Start with ~500–1000 development examples if feasible.

But your mandatory golden set is separate.

10. Golden evaluation set

You need:

150–250 hand-labelled examples

I'd target:

200 examples

Make it stratified.

For example:

Intent A → 15
Intent B → 15
Intent C → 15
...

Also intentionally include difficult cases.

Examples:

ambiguous wording
multi-intent messages
very short messages
angry customers
missing context
follow-up messages
high-risk payment issues

The golden set must be held out from:

training
RAG index
prompt examples

Otherwise you introduce leakage.

11. Golden set schema

I'd use:

example_id
customer_message
intent
intent_notes
expected_action
reference_response
difficulty

For example:

001
"My package was due yesterday..."
late_delivery
Clear delivery delay
auto_handle
"Sorry for the delay..."
medium

Another:

002
"I was charged twice for my order..."
payment_issue
Requires account/payment verification
escalate
"Please contact support..."
high
12. Phase 7 — Build the historical resolution KB

This is the heart of the project.

Create:

amazonhelp_cases.csv

Suggested schema:

case_id
conversation_id
customer_problem
conversation
amazon_responses
resolution
intent
resolution_quality
turn_count
13. What exactly is resolution?

Don't invent it.

For every conversation, extract what AmazonHelp actually did.

Example:

Customer:
"My package is late."

AmazonHelp:
"Sorry about the delay. Please check your order status."

Resolution:

"AmazonHelp directed the customer to check the order status and provided support guidance."

Another:

Customer:
"I received the wrong product."

AmazonHelp:
"Please contact support so we can help with the order."

Resolution:

"AmazonHelp directed the customer to customer support for assistance with the incorrect item."

Notice we're describing the historical action.

We're not saying:

"Amazon will send a replacement."

unless the historical conversation actually supports that.

14. Resolution quality

Add:

resolution_quality

Possible values:

strong
medium
weak
Strong

Clear problem + clear handling.

Medium

Useful support direction but no clear final resolution.

Weak

Minimal/ambiguous response.

This prevents the RAG system from treating:

"Please DM us"

as equally valuable as a detailed historical resolution.

15. Phase 8 — RAG document construction

For every case, create an embedding document.

For example:

Customer problem:
My package was supposed to arrive yesterday but hasn't arrived.

Historical AmazonHelp handling:
AmazonHelp apologized for the delay, directed the customer to check
the order status, and provided support guidance.

Intent:
late_delivery

Store metadata:

{
    "case_id": "C001",
    "intent": "late_delivery",
    "resolution_quality": "strong",
    "turn_count": 4
}
16. Don't embed individual tweets

This is an important design decision.

Bad approach:

Tweet 1 → embedding
Tweet 2 → embedding
Tweet 3 → embedding
...

That loses the support context.

Better:

Historical case
    ↓
Customer problem + historical handling
    ↓
One embedding

Then retain the original conversation as metadata/evidence.

17. Phase 9 — Embedding model

Use something lightweight and reproducible.

For example:

sentence-transformers

with a small general-purpose embedding model.

The goal isn't to spend hours building infrastructure.

Something like:

model.encode(case_documents)

Then create a FAISS index.

18. Phase 10 — Retrieval baseline

Before fancy retrieval, establish a baseline.

Baseline 1 — TF-IDF

For an incoming message:

query
 ↓
TF-IDF
 ↓
cosine similarity
 ↓
top K cases

This gives you a strong simple baseline.

Baseline 2 — Embedding retrieval
query
 ↓
embedding
 ↓
FAISS
 ↓
top K

Then compare.

19. Phase 11 — Intent-aware retrieval

Once we have an intent prediction:

Query:

"My package is late."

Intent:

late_delivery

Retrieve candidates based on semantic similarity, but incorporate intent.

For example:

final_score = (
    0.70 * semantic_similarity
    + 0.20 * intent_match
    + 0.10 * resolution_quality
)

These weights should be treated as tunable, not sacred.

You can evaluate a few settings on the development set.

20. Phase 12 — Retrieval evaluation

You need to demonstrate whether RAG actually retrieves useful historical cases.

Metrics:

Recall@K
Recall@1
Recall@3
Recall@5
Recall@10
MRR

Mean Reciprocal Rank.

If the correct historical handling appears near the top, MRR improves.

Also manually inspect:

Query
Top 3 retrieved cases
Relevant/not relevant

This gives you qualitative evidence.

21. Phase 13 — Intent classifier

You need a classifier.

I'd implement two baselines.

Baseline A — Majority class

Predict:

most_common_intent

This is intentionally trivial.

Baseline B — TF-IDF + Logistic Regression
text
 ↓
TF-IDF
 ↓
Logistic Regression
 ↓
intent

This is your simple ML baseline.

22. Main intent model

You have two reasonable options.

Option A — LLM classifier

Prompt:

Classify this customer message into exactly one of these intents.

Intent definitions:
...

Customer message:
...

Return JSON:
{
  "intent": "...",
  "confidence": 0-1
}

This is easy and potentially strong.

Option B — embedding classifier

Use embeddings + nearest labelled examples.

I'd probably test both and select based on golden-set performance and runtime.

23. Phase 14 — Response generation

The LLM receives:

Customer message

Predicted intent

Retrieved historical cases

Routing constraints

Prompt should explicitly say:

You are a customer support response assistant.

Use historical AmazonHelp cases as evidence.

Do not invent:
- refunds
- compensation
- policies
- delivery dates
- account actions
- guarantees

If the issue requires account-specific investigation,
recommend escalation instead.

Return:
1. response
2. evidence_case_ids
3. escalation_recommendation
4. reason
24. Example generation

Input:

Customer:
"My package is late."

Retrieved:

C001
C017
C043

LLM:

{
  "response": "Sorry about the delay. Please check your order status through your Amazon account. If the order is still delayed, customer support can help investigate it.",
  "evidence_case_ids": ["C001", "C017"],
  "escalate": false,
  "reason": "Low-risk delivery issue with strong historical support examples."
}
25. Phase 15 — Escalation logic

Don't let the LLM arbitrarily decide everything.

Create explicit routing rules.

For example:

ESCALATE if:

1. High-risk intent
2. Account-specific action required
3. Payment/refund verification required
4. Very low intent confidence
5. Poor retrieval evidence
6. Conflicting retrieved cases
7. No historically relevant case
8. Customer requests something outside known support handling
9. Sensitive/private information is required

Otherwise:

AUTO-HANDLE
26. Risk score

You can implement:

risk_score = (
    intent_risk
    + low_confidence_risk
    + retrieval_risk
    + account_specific_risk
)

Then:

risk_score >= threshold
        ↓
ESCALATE

otherwise
        ↓
AUTO-HANDLE

Tune the threshold on the development set.

27. Very important: optimize for false auto-handling

For customer support, this:

Wrongly auto-handle a refund/payment issue

is worse than:

Escalate a simple delivery question

So your routing evaluation should emphasize:

False Auto-Handle Rate

not just accuracy.

28. Phase 16 — Reply evaluation

This is another major part of the assignment.

Don't evaluate generated responses only with BLEU/ROUGE.

They don't make much sense for support responses.

Use an LLM judge with a structured rubric.

29. LLM judge rubric

For every generated answer, score:

1. Groundedness

Is the answer supported by historical evidence?

1–5
2. Correctness

Does it correctly address the customer's issue?

1–5
3. Resolution alignment

Does it follow how AmazonHelp historically handled similar issues?

1–5
4. Helpfulness

Does the customer know what to do next?

1–5
5. Safety

Does it avoid unsupported promises/account actions?

1–5
6. Escalation appropriateness

Was escalation recommended when necessary?

1–5
30. Human agreement

The assignment specifically asks for human agreement evidence.

Take approximately:

50 generated responses

Have humans score them using the same rubric.

Then compare:

Human score
vs
LLM judge score

Report something like:

Agreement rate
Spearman correlation
or Cohen's kappa for categorical decisions

Don't fabricate these numbers—you'll calculate them from your actual annotations.

31. Phase 17 — Full evaluation harness

Build one command:

python -m src.evaluation.run

It should produce something like:

========================================
INTENT CLASSIFICATION
========================================

Majority baseline
Accuracy: ...
Macro F1: ...

TF-IDF + Logistic Regression
Accuracy: ...
Macro F1: ...

LLM classifier
Accuracy: ...
Macro F1: ...


========================================
RETRIEVAL
========================================

Recall@1: ...
Recall@3: ...
Recall@5: ...
MRR: ...


========================================
RESPONSE QUALITY
========================================

Groundedness: ...
Correctness: ...
Resolution alignment: ...
Helpfulness: ...
Safety: ...


========================================
ROUTING
========================================

Auto-handle precision: ...
Escalation recall: ...
False auto-handle rate: ...


========================================

This is extremely valuable for the assignment.

32. Phase 18 — Compare against trivial/simple baselines

Your report needs this.

I'd use:

System	Intent	Retrieval	Response
Majority	Majority intent	—	—
TF-IDF	TF-IDF + LR	TF-IDF	—
Semantic RAG	Embedding retrieval	FAISS	LLM
Intent-aware RAG	LLM/ML intent + FAISS	Reranked	LLM
Final	Intent + RAG + routing	Reranked	Grounded LLM

You don't necessarily need all combinations if time is limited, but at minimum show that each important component adds value.

33. Ablation study

This will make your submission much stronger.

Test:

A. No RAG
LLM → response
B. RAG
LLM + historical cases
C. RAG + intent filtering
LLM + intent + historical cases
D. RAG + intent + quality reranking
LLM + intent + reranking

Then compare response quality.

You can say:

Adding historical support cases improved resolution alignment, while intent-aware retrieval reduced irrelevant retrieved examples.

Only if your actual results demonstrate it.

34. Phase 19 — Top 5 failure analysis

The assignment explicitly asks for this.

For each failure:

Input
Predicted intent
Expected intent
Retrieved cases
Generated response
Expected behavior
Why it failed
Hypothesis
Potential fix

Example:

Failure #1 — Intent overlap
Message:
"My package disappeared after it said delivered."

Predicted:
late_delivery

Expected:
missing_package

Why:
Both intents contain delivery-related language.

Hypothesis:
Classifier relies too heavily on lexical similarity.

Fix:
Add hard negative examples distinguishing delayed vs delivered-but-missing.

This is much stronger than simply saying:

"The model made a mistake."

35. Phase 20 — "What is misleading about my headline number?"

This is a mandatory section.

Suppose your headline says:

92% intent accuracy

You should explicitly acknowledge:

1. Golden set is relatively small.
2. Distribution may not match real production traffic.
3. Rare intents have fewer examples.
4. Intent accuracy doesn't guarantee good responses.
5. Safe routing can still fail even when classification is correct.
6. The historical dataset contains only observed AmazonHelp behavior.
7. RAG quality depends on whether a similar historical case exists.

This demonstrates maturity.

36. Phase 21 — Demo pipeline

Build a simple CLI.

For example:

python -m src.pipeline

Input:

Customer message:
"My order was supposed to arrive yesterday."

Output:

Intent
------
late_delivery

Confidence
----------
0.91

Retrieved historical cases
--------------------------
C001
C017
C043

Draft response
--------------
Sorry about the delay. Please check your order status...

Decision
--------
AUTO-HANDLE

Reason
------
Low-risk delivery issue with strong historical support evidence.

This is enough.

You don't need a fancy frontend.

37. Phase 22 — Reproducibility

The evaluator should ideally be able to do:

git clone ...
cd hiver-ai-support-agent

pip install -r requirements.txt

python -m src.evaluation.run

and finish in:

< 15 minutes

Therefore:

don't require the full 3M tweet dataset
provide a processed subset
don't train huge models
don't build distributed infrastructure
don't require GPUs
keep LLM calls limited
cache embeddings
cache LLM responses during evaluation
38. Caching

This is particularly important if you're using an API LLM.

Create:

artifacts/
└── llm_cache.json

For each evaluation input:

hash(prompt)
     ↓
cache lookup
     ↓
if exists → reuse
else → API call

This makes your evaluation reproducible and prevents unnecessary API calls.

39. Suggested implementation order

Don't implement everything simultaneously.

Follow this exact order:

Day/Phase 1 — Data
✓ Load both CSVs
✓ Analyze dataset
✓ Verify conversations
✓ PII sanitization
✓ Quality filtering

↓

Phase 2 — Intent
✓ Explore intents
✓ Define taxonomy
✓ Label development data
✓ Create golden set
✓ Majority baseline
✓ TF-IDF + LR
✓ Main classifier

↓

Phase 3 — Historical KB
✓ Extract resolution
✓ Assign quality
✓ Create amazonhelp_cases.csv
✓ Create embedding documents
✓ FAISS index

↓

Phase 4 — RAG
✓ Semantic retrieval
✓ Intent-aware retrieval
✓ Reranking
✓ Retrieval evaluation

↓

Phase 5 — Generation
✓ Prompt
✓ Evidence-based response
✓ Structured output
✓ LLM judge

↓

Phase 6 — Routing
✓ Risk rules
✓ Escalation logic
✓ False-auto-handle evaluation

↓

Phase 7 — Final evaluation
✓ Baselines
✓ Ablations
✓ LLM judge
✓ Human agreement
✓ Confusion matrix
✓ Failure analysis

↓

Phase 8 — Submission
✓ README
✓ 6-page report
✓ Decision log
✓ Reproduction command
✓ Results
✓ Top 5 failures
✓ "Misleading headline number"
40. The most important design decision

I'd keep the final system conceptually like this:

                    CUSTOMER MESSAGE
                           │
                           ▼
                  ┌─────────────────┐
                  │ Intent Classifier│
                  └────────┬────────┘
                           │
                           ▼
                    predicted intent
                           │
                           ▼
                  ┌─────────────────┐
                  │ Historical RAG  │
                  │     Search      │
                  └────────┬────────┘
                           │
                       top cases
                           │
                           ▼
                  ┌─────────────────┐
                  │ Evidence/Risk   │
                  │     Check       │
                  └────────┬────────┘
                           │
                ┌──────────┴──────────┐
                ▼                     ▼
             SAFE                    RISKY
                │                     │
                ▼                     ▼
        ┌───────────────┐       ┌────────────┐
        │ Generate Reply│       │ ESCALATE   │
        └───────┬───────┘       └────────────┘
                │
                ▼
        ┌───────────────┐
        │ Grounded Reply│
        └───────────────┘

That's a very defendable architecture in an interview.

You can explain every component, every metric, and every trade-off.

What I would NOT build

For this assignment, explicitly avoid:

❌ Twitter API integration
❌ Real customer accounts
❌ Autonomous refunds
❌ Autonomous order modifications
❌ Multi-agent architecture
❌ LangGraph/CrewAI orchestration
❌ Fine-tuning a large model
❌ Complex frontend
❌ Production Kubernetes
❌ Real-time Kafka pipeline
❌ Multilingual support
❌ Voice support

Those things consume time without directly improving the evaluation evidence.

Your immediate next 3 tasks

Since you've already reconstructed the two CSVs, don't jump into FAISS yet.

Do these next:

1. Analyze amazonhelp_conversations.csv
          ↓
2. Clean + sanitize + quality-filter conversations
          ↓
3. Define the intent taxonomy from the actual data

Only after that should we create amazonhelp_cases.csv.

That order matters because the intent taxonomy and resolution quality will determine how we construct the RAG KB and, ultimately, how we evaluate it.