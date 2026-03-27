# MITR — Research Engineering Intern Take-Home

## Background

This repository contains the code and results for **MITR (Mutual Information Transformer Regularization)**, a technique that penalizes redundancy between consecutive transformer layers to improve logical reasoning.

The core idea: if two adjacent layers learn the same thing, that's wasted capacity. By adding a mutual-information penalty, we force each layer to contribute something distinct — which should help with multi-step logical consistency.

### What's in this repo

| File | Description |
|------|-------------|
| `results.md` | DistilBERT experiment results — 4 MI strategies compared on BoolQ |
| `roberta_bert_results.md` | BERT and RoBERTa experiment results — scaling MITR to 12-layer models |
| `cka_results.png` | Chart for the DistilBERT experiments |
| `roberta_results.png` | Chart for the BERT/RoBERTa experiments |
| `mitr_distilbert_boolq.ipynb` | Full training notebook — DistilBERT on BoolQ with all MI strategies |
| `mitr_bert_roberta_boolq.ipynb` | Full training notebook — BERT and RoBERTa on BoolQ |

Both notebooks are self-contained: they install dependencies via pip and define all model code inline. They are designed to run on Google Colab with a GPU runtime.

---

## Take-Home Evaluation

**Time:** 2.5 hours (hard limit — submit whatever you have)

**Tools:** You may use any LLM (ChatGPT, Claude, Copilot, etc.) freely. Using AI tools well is an engineering skill we value. You are fully responsible for the correctness of everything you submit.

**Submit:** (1) One PDF or markdown document with Parts A–E, and (2) one runnable Colab notebook for Part D. Commit everything to your fork.

**Setup:** Fork this repo. Read the two results markdown files and look at the two PNG charts. Skim at least one notebook. Then start your timer.

**Tip:** Part D requires GPU training time. Start it early and work on Part E while it runs.

---

### Part A: Critical Review (25 minutes · 25 points)

You've inherited this codebase and need to take it to a workshop submission. Before adding anything, you need to know what's missing.

**Review the repository — code, notebooks, results files, and images. Identify the 3 most critical weaknesses that would cause a reviewer to reject this work.**

For each weakness, write:

| | |
|---|---|
| **1. What is the problem?** | Reference specific files, numbers, or code. |
| **2. Why would a reviewer reject over this?** | One sentence. |
| **3. How would you fix it?** | Name the specific experiment, analysis, or revision. |

> We are looking for issues that reflect genuine engagement with *this specific* codebase and results, not generic ML advice ("needs more data"). Prioritization matters — there are many minor issues, but which 3 would actually sink the paper?

---

### Part B: Workshop Selection (15 minutes · 10 points)

**Choose one workshop at a top-tier venue (ICLR, NeurIPS, ICML, ACL, EMNLP 2025 or 2026) where you would submit this work.**

1. Name the specific workshop (full name + venue)
2. In 150 words or fewer, pitch why this work fits that workshop's scope
3. Name 2 published papers from that workshop (or its previous editions) that this work directly relates to. For each, state the relationship in one sentence.

> A well-justified niche workshop beats a vaguely justified flagship. We're testing whether you understand where this work sits in the field.

---

### Part C: Experimental Design (30 minutes · 25 points)

**Propose exactly 3 new experiments that would strengthen this submission. Rank them 1 (most important) to 3.**

For each experiment:

| Field | Your answer |
|-------|-------------|
| **Name** | (short label) |
| **Hypothesis** | "If [we do X], then [Y will happen], because [Z]." |
| **Expected result** | What numbers/trends would confirm the hypothesis? |
| **What a negative result means** | If the hypothesis is wrong, what does that tell us? |
| **Estimated GPU hours** (Colab T4) | |
| **How this changes the paper's central claim** | One sentence. |

**Then in 100 words or fewer:** Justify your ranking. Why is #1 the most important experiment to run first?

> We value experiments that address genuine gaps in the current work over safe "run on more data" proposals. Your hypotheses must be falsifiable. Bonus: if experiment #1 directly addresses a weakness from Part A.

---

### Part D: Implementation (60 minutes · 30 points)

**Implement and run your #1 experiment from Part C.**

Starting from one of the existing notebooks (`mitr_distilbert_boolq.ipynb` or `mitr_bert_roberta_boolq.ipynb`), create a Colab notebook that:

1. Runs your proposed experiment
2. Produces at least one new chart or table of results
3. Includes a 150-word (max) discussion: what do the results show, and do they support your hypothesis?

**Practical guidance:**
- You may reduce epochs (e.g., 3 instead of 5) or use a data subset to fit Colab's compute. If you scope down, state what the full experiment would look like and what you'd expect.
- If your #1 experiment isn't feasible on Colab (e.g., requires A100-only memory), implement #2 or #3 instead and briefly explain why you pivoted.
- The notebook must run end-to-end without manual intervention on a Colab GPU runtime.
- **Start training early** — work on Part E while it runs.

> This is the most heavily weighted section. A notebook that runs and produces interpretable results with an honest discussion (including negative results) will always score higher than an ambitious notebook that crashes.

---

### Part E: Abstract (10 minutes · 10 points)

**Write a 200-word (max) abstract for the workshop paper, incorporating the results from Part D.**

Your abstract must contain:
1. The problem (one sentence)
2. The method — MITR — in one sentence
3. One central claim (not a list of everything — the single most important finding)
4. One quantitative result that supports the claim

> A great abstract makes a reader understand what is new and why it matters. A weak abstract describes what was done without staking a claim.

---

### What We're Looking For

This isn't a trick test. Here's what separates levels:

| Level | Signal |
|-------|--------|
| **Strong hire** | Part A identifies issues specific to this repo (not generic ML critique). Part C experiments are creative and clearly motivated by Part A gaps. Part D notebook runs cleanly and discussion is honest about what was found. Abstract stakes a clear, defensible claim. |
| **Solid** | Parts A–C are reasonable and well-written. Part D runs but experiment is safe (e.g., just more epochs). Abstract is competent. |
| **Concerning** | Part A issues are generic ("needs more data"). Part D notebook has errors. Abstract claims MITR universally improves everything (contradicted by the BERT results in the repo). |

We care about **judgment** (did you prioritize the right things?), **honesty** (did you acknowledge what the results don't show?), and **execution** (does the code run?). Polished writing is a plus but never compensates for broken code or unsupported claims.

---

*Good luck. Start with `results.md`, `roberta_bert_results.md`, and the two `.png` files — those contain the full story.*
