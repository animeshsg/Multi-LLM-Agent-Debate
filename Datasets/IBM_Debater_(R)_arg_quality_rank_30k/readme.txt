NAME: IBM Debater(R) - IBM-ArgQ-Rank-30kArgs

VERSION: v1

RELEASE DATE: November 25, 2019

DATASET OVERVIEW

30,497 arguments for 71 topics labeled for quality and stance, split into train, dev and test sets.

The dataset includes: 
1. One CSV file containing all arguments collected and labeled for quality and stance, for train, dev and test sets.

The datasets are released under the following licensing and copyright terms:
• (c) Copyright Wikipedia (https://en.wikipedia.org/wiki/Wikipedia:Copyrights#Reusers.27_rights_and_obligations)
• (c) Copyright IBM 2014. Released under CC-BY-SA (http://creativecommons.org/licenses/by-sa/3.0/)

CONTENTS

One CSV file, arg_quality_rank_30k.csv, contain the following columns for each sentence:
1. argument
2. topic - the topic context of the argument
3. set - either train, dev or test
4. WA - the quality label according to the weighted-average scoring function
5. MACE-P - the quality label according to the MACE-P scoring function
6. stance_WA - the stance label according to the weighted-average scoring function
7. stance_WA_conf - the confidence in the stance label according to the weighted-average scoring function

Quality labels
--------------
For an explanation of the quality labels presented in columns WA and MACE-P, please see section 4 in note (1).

Stance labels
-------------
There were three possible annotations for the stance task: 1 (pro), -1 (con) and 0 (neutral). The stance_WA_conf column refers to the weighted-average score of the winning label. The stance_WA column refers to the winning stance label itself.

NOTES:
(1) Please cite: 

    A Large-scale Dataset for Argument Quality Ranking: Construction and Analysis
    Shai Gretz, Roni Friedman, Edo Cohen-Karlik, Assaf Toledo, Dan Lahav, Ranit Aharonov and Noam Slonim
    AAAI 2020