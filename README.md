# A Minimal Model of L3-to-L1 Reverse Transfer in Trilingual Word Recognition

## Overview
This repository contains a minimal computational model simulating how learning Japanese (L3) affects Chinese (L1) word recognition. The model is based on the lexical competition and language node inhibition mechanisms in the BIA+ model (Dijkstra & Van Heuven, 2002).

## Scientific Question
When a native Chinese speaker learning Japanese encounters a Japanese-Chinese homograph (e.g., 大学), judging "whether it is a Chinese word" becomes more difficult. This model simulates the underlying mechanism: after L3 learning, increased inhibition from L3 lexical nodes onto L1 nodes leads to a decrease in L1 activation.

## Model Design
- **Three lexical nodes**: L1 (Chinese), L2 (English), L3 (Japanese)
- **Two scenarios**:
  - **Scenario A**: Before L3 learning (L3 baseline = 0.1)
  - **Scenario B**: After L3 learning (L3 baseline = 0.4, inhibition weight = 1.8)
- **Key parameters**: activation baseline, decay rate (0.1), mutual inhibition strength (0.08)

## Key Result
The model shows that after L3 learning, the enhanced L3 node exerts stronger competitive inhibition on the L1 node. This computationally captures the mechanism behind the behavioral phenomenon: L3 lexical activation suppresses L1 processing, making homograph judgment more difficult.

## How to Run
1. Clone this repository
2. Install dependencies: `pip install numpy matplotlib`
3. Run: `python reverse_transfer_model.py`

## Reference
Dijkstra, T., & Van Heuven, W. J. B. (2002). The architecture of the bilingual word recognition system: From identification to decision. *Bilingualism: Language and Cognition*, 5(3), 175–197.

## Author
[Your Name] — Independent Researcher
SSCI Q1 First Author & Corresponding Author
