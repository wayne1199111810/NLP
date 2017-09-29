# Collocation Identification
Using the subset of Treebnak dataset, I measure the top 20 collocations from the bigram with two different correlation measure: chi-square and pointwise mutual information (PMI)

## Assumption
* Unigrams and bigrams that include tokens consisting only of punctuation are not included
* It discards bigrams that occurs less than 5 times.

## Top 20 Bigrams Along With Chi-Square Scores
|      W1     |       W2      |    Value   |
| ----------- | :-----------: |:----------:|
|Negotiable   |  bank-backed  |    162587.0|
|Prebon       |  U.S.A        |    162587.0|
|Hells        |  Angels       |    162587.0|
|Witter       |  Reynolds     |    162587.0|
|Fulton       |  Prebon       |    162587.0|
|avoid        |  default      |    162587.0|
|grand        |  jury         |    162587.0|
|Partners     |  Limited      |    162587.0|
|Easy         |  Eggs         |    162587.0|
|fetal-tissue |  transplants  |    162587.0|
|LATE         |  EURODOLLARS  |    162587.0|
|OF           |  DEPOSIT      |    162587.0|
|CALL         |  MONEY        |    162587.0|
|Waertsilae   |  Marine       |    162587.0|
|Stocks       |  Volume       |    162587.0|
|Houston      |  Lighting     |    162587.0|
|MGM          |  Grand        |    162587.0|
|instruments  |  typically    |    162587.0|
|Louisville   |  Ky.          |    162587.0|
|Corporate    |  Issues       |    162587.0|
|Wilmington   |  Del.         |    162587.0|

## Top 20 bigrams along with their PMI scores
|     W1     |      W2     |      Value    |
| ---------- |:-----------:|:-------------:|
| Negotiable | bank-backed | 17.7440480074 |
| Prebon     | U.S.A       | 17.7440480074 |
| MERRILL    | LYNCH       | 17.7440480074 |
| INTERBANK  | OFFERED     | 17.7440480074 |
| Fulton     | Prebon      | 17.7440480074 |
| TREASURY   | BILLS       | 17.7440480074 |
| BANKERS    | ACCEPTANCES | 17.7440480074 |
| LATE       | EURODOLLARS | 17.7440480074 |
| LYNCH      | READY       | 17.7440480074 |
| ASSETS     | TRUST       | 17.7440480074 |
| Wastewater | Treatment   | 17.7440480074 |
| READY      | ASSETS      | 17.7440480074 |
| Zoete      | Wedd        | 17.7440480074 |
| Larsen     | Toubro      | 17.4810136015 |
| Easy       | Eggs        | 17.4810136015 |
| OF         | DEPOSIT     | 17.4810136015 |
| Hang       | Seng        | 17.4810136015 |
| Deb        | Shops       | 17.4810136015 |
| Bare-Faced | Messiah     | 17.4810136015 |
| CALL       | MONEY       | 17.4810136015 |
| HOME       | LOAN        | 17.4810136015 |

## Discussion of which of the two measures works better to identify collocations, based on analysis of the top 20 bigrams produced by each measure

The result of the chi-square is exactly the same as the number of total bigram. It can be shown that if E(1, 2) and E(2, 1) equal to zero (discard bigram less than 5), the chi-square will be the number of total bigram. From this observation, the top 20 bigrams from the chi-square might just be different owing to the sorting method. In contrast, PMI does not restrict to this phenomenon. It also provides the different result for the measure of correlation. Consequently, I think PMI works better on this task.