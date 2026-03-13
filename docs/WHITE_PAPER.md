# Guiding Graph Learning with Interpretable Substructure Encodings

This document is the conceptual background for the standalone `abstractgraph`
core package and the split ecosystem around it.

Package mapping:
- `abstractgraph`: representation, operators, hashing, display, vectorization
- `abstractgraph-ml`: estimators and analysis on top of those representations
- `abstractgraph-generative`: rewriting and generation on top of those
  representations

## Executive Summary

Modern machine learning models can be accurate but remain difficult to inspect, debug, and correct at the mechanism level. This white paper introduces **Abstract Graphs** as a mechanistic intermediate representation (IR): a representation layer between raw structured data and predictive models where each variable corresponds to an explicit, inspectable subgraph predicate.

Rather than relying only on latent parameters, Abstract Graphs provide stable, composable, human-meaningful entities that support diagnosis, intervention, and iterative refinement.

## Table of Contents

1. [Motivation: Why Machine Learning Needs an Abstraction Layer](#1-motivation-why-machine-learning-needs-an-abstraction-layer)
2. [Definition of an Abstract Graph](#2-definition-of-an-abstract-graph)
3. [What the Abstract Graph Represents](#3-what-the-abstract-graph-represents)
4. [Pharmacophore Example (Chemoinformatics)](#4-pharmacophore-example-chemoinformatics)
5. [Relation to Existing Molecular Descriptors](#5-relation-to-existing-molecular-descriptors)
6. [Abstraction in Computer Science](#6-abstraction-in-computer-science)
7. [Causal Abstraction](#7-causal-abstraction)
8. [Category-Theoretic Interpretation](#8-category-theoretic-interpretation)
9. [Information-Theoretic Interpretation](#9-information-theoretic-interpretation)
10. [Learning as Search Over Vocabularies](#10-learning-as-search-over-vocabularies)
11. [Human Reasoning and Compositionality](#11-human-reasoning-and-compositionality)
12. [Model Selection Criteria](#12-model-selection-criteria)
13. [Conceptual Interpretation](#13-conceptual-interpretation)
14. [Central Research Question](#14-central-research-question)

## 1. Motivation: Why Machine Learning Needs an Abstraction Layer

Modern machine learning models are effective predictors but poor scientific objects.
They produce outputs, yet they do not expose mechanisms.

Typical workflow:

1. Train model
2. Evaluate performance
3. Observe failures
4. Retrain

The missing step is:

> Diagnose why the model failed.

The core difficulty is structural:

- Neural networks represent knowledge as distributed parameters.
- Latent variables are not stable referents.
- A user cannot intervene locally.

Representation learning explicitly aims to discover useful features automatically, but these features are internal and uninterpretable.

Reference:
Bengio, Courville, Vincent (2013), Representation Learning.
https://arxiv.org/abs/1206.5538

The problem is therefore not accuracy.

It is lack of operational interface.

A human cannot:

- Inspect a concept.
- Alter a concept.
- Test a hypothesis about the model.

The proposal behind Abstract Graphs is:

> Machine learning needs an intermediate representation that exposes mechanisms rather than parameters.

## 2. Definition of an Abstract Graph

An **Abstract Graph** consists of two related structures.

### Pre-image Graph (G)

The concrete system, for example:

- Molecule
- Sentence
- Biological network
- Relational database

### Image Graph (H)

A graph whose nodes represent **subgraph predicates of G**.

Formally:

$$
\pi : \mathcal{P}(G) \rightarrow V(H)
$$

where $\mathcal{P}(G)$ is a selected family of subgraphs.

A node in $H$ corresponds to a pattern in $G$, not an element.

Examples:

- Neighborhood of radius 1
- Two neighborhoods at bounded distance
- Node participating in aromatic cycle
- Relational event in a story

### Important Property

The image graph is not necessarily a simplification.

$$
|V(H)| \geq |V(G)|
$$

The image graph can be larger because it makes hidden relational structure explicit.

## 3. What the Abstract Graph Represents

The Abstract Graph is a **space of observables**.

Instead of describing a system by its parts, we describe it by the patterns detectable within it.

| Description Type | Example |
| --- | --- |
| Object-based | Atom A |
| Relational | Atom A hydrogen-bonds with ring B |

Humans reason in relational descriptions, not raw states.

The Abstract Graph therefore converts:

> structure -> named relational mechanisms

## 4. Pharmacophore Example (Chemoinformatics)

A pharmacophore is a spatial arrangement of features responsible for biological activity.

IUPAC definition:
https://doi.org/10.1351/goldbook.P04524

Example motif:

- Hydrogen bond donor
- Five bonds away from aromatic ring

This is naturally expressible as a subgraph predicate over the molecular graph.

In the Abstract Graph, the entire motif becomes a single node.

This node can be:

- Counted
- Compared
- Linked to activity

Thus predictive modeling becomes:

$$
\text{bioactivity} = f(\text{mechanisms})
$$

not

$$
\text{bioactivity} = f(\text{atoms})
$$

## 5. Relation to Existing Molecular Descriptors

### Morgan Fingerprints

Enumerate local neighborhoods and hash them into bit vectors.

Rogers and Hahn (2010):
https://doi.org/10.1021/ci100050t

Limitation:

- Features exist.
- Meaning is lost.

### Graph Neural Networks

Learn features automatically.

Limitation:

- No symbolic identity.
- No inspectable mechanism.

### Abstract Graphs

Provide:

- Explicit predicates
- Named structures
- Compositional relations

Key distinction:

| Method | Representation |
| --- | --- |
| Fingerprint | Hashed pattern |
| GNN | Latent vector |
| Abstract Graph | Interpretable mechanism |

## 6. Abstraction in Computer Science

Software engineering manages complexity through abstraction barriers.

Users interact with operations, not implementation.

Primary reference:
Abelson and Sussman, Structure and Interpretation of Computer Programs.
https://web.mit.edu/6.001/6.037/sicp.pdf

Abstract Graph nodes behave similarly.

A node is an interface:

> a donor near aromatic ring

The internal molecular details are hidden.

## 7. Causal Abstraction

A correct abstraction preserves causal relationships across levels.

Beckers and Halpern (2021):
https://arxiv.org/abs/2106.02997

Mapping:

| Level | Meaning |
| --- | --- |
| Pre-image | Micro-variables |
| Image graph | Macro-variables |

Therefore model failure becomes localized to a mechanism.

Instead of retraining blindly, one can modify the representation.

## 8. Category-Theoretic Interpretation

The Abstract Graph transforms an object into its observables.

Instead of defining a graph only as nodes and edges, we can treat it as a family of relational patterns appearing in it.

This resembles representing an object by how it behaves under interactions.

References:
Mac Lane, Categories for the Working Mathematician.
https://www.sas.rochester.edu/mth/sites/doug-ravenel/otherpapers/maclanecat.pdf

Spivak, Seven Sketches in Compositionality.
https://dspivak.net/7Sketches.pdf

Interpretation:

The Abstract Graph is a structured catalog of observable behaviors.

## 9. Information-Theoretic Interpretation

A good abstraction preserves relevant information.

Information Bottleneck principle:
retain information about target $Y$ while compressing input $X$.

Exposition:
https://arxiv.org/abs/1503.02406

Abstract Graph nodes correspond to functional equivalence classes of structures.

Different graphs that act the same biologically can map to the same abstract pattern.

## 10. Learning as Search Over Vocabularies

Traditional ML optimizes parameters.

Abstract Graph learning optimizes concept vocabulary.

This resembles predicate invention in ILP.

Muggleton (1991):
https://doi.org/10.1016/0004-3702(91)90035-U

We are discovering what entities exist in the explanatory model.

## 11. Human Reasoning and Compositionality

Humans reason with compositional symbols.

Overview:
Lake et al. (2017):
https://doi.org/10.1017/S0140525X16001837

Abstract Graphs provide:

- Stable referents
- Composable relations
- Interpretable explanations

## 12. Model Selection Criteria

Two objectives:

### Utility

Predictive performance.

### Simplicity

Short decomposition function.

This mirrors Occam-style generalization control (structural risk minimization; Vapnik).

Interpretation:

We minimize the description length of the explanation.

## 13. Conceptual Interpretation

Abstract Graphs are best understood as a mechanistic intermediate representation for machine learning.

Analogy:

| Computing | Abstract Graphs |
| --- | --- |
| Machine code | Pre-image graph |
| Intermediate representation | Abstract graph |
| Program reasoning | Human interpretation |

## 14. Central Research Question

The main open question is:

> Which families of subgraph predicates produce stable, causal, human-meaningful mechanisms?

Correct abstraction yields:

- Interpretability
- Local debugging
- Guided correction

Incorrect abstraction yields:

- Misleading explanations

Therefore, the framework is a formal method for constructing interpretable variables from structured data.
