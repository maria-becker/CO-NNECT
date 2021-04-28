Establishing Commonsense Knowledge Paths between Concepts from Sentences Via Forward Chaining

This readme explains the code for applying COMET (Bosselut et al., 2019) for forward chaining, as described in our paper (Becker et al. 2021). This forward chaining procedure is part of our framework CO-NNECT, which we propose for enriching texts with commonsense knowledge in the form of high-quality single- and multi-hop knowledge paths between concepts in texts. With the following code, you can generate single- and multihop commonsense knowledge paths between concepts from two sentences, via target prediction. The code for extracting concepts from texts can be found in our CoCo-Ex repository. The code for generating direct relations between concepts from texts, using our relation classification model, can be found in the COREC-LM directory of this repository.

Running the code requires the following software components:

- Python 3.6/3.7
- spacy 2.3.5
- nltk 3.5
- gensim 3.8.3
- pandas 1.2
- stanford parser 3.9.2

Step 1: Extract concepts from a given pair of sentences using CoCo-Ex (https://github.com/Heidelberg-NLP/CoCo-Ex)

Step 2: Run pipeline.py with the following parameters:

-inputfile: tsv file with pairs of concepts, one pair per line, separated by tabs. You can generate those input pairs with our concept extraction tool CoCo-Ex,  (just use the output from CoCo-Ex as input to the model), or create your own input file.

-experiment_name: The experiment_name is used to name temporary output files (but not the final outputs). If you start several forwardchaining runs in parallel, it is very important that the experiment_name is distinct for each of them.

-out: The directory (e.g. paths /) to which the results should be written. The path must end with “/”. The output file name is then [-out] + [name of the input file] + [configurations]

-sim: Similarity metric for comparing generated concepts to target concepts, default: cosim ("cos")

-emb: Path to embeddings which are used for computing similarity, default: “numberbatch-en-17.06.txt” (in the main directory). 

-hops: number of hops (any integer), default: 3 

-beams: beam size (number of generated target concepts per source concept-relation pair, default: 10 

-contextsimthresh: This is the threshold that is applied for comparing generated concepts to target concepts during path generation. We recommend 0.7.

-keep: Either “all” or any integer smaller than the beam size. Decides how many of the predictions (over the threshold) from the beam should be used for path generation, default: “all”.

-threshold: This is the threshold that is applied for comparing generated concepts to target concepts for path termination. We recommend 0.95.

-pos_filter: Flag, which turns PoS-Filter on. It is a type-based PoS sequence filtering, where the type
is dependent on the predicted relation. We highly recommend setting the flag.

-lemma_check: Flag which checks if the source and target tokens of a path (tokens and lemmatized tokens) are identical (and only consider paths where this is not the case). We highly recommend setting the flag.

-reverse: Flag which specifies which pretrained model is used. If you set the flag, it will use the COMET model that we pretrained on CN-13, including reverse triples. If you don’t set the flag, it will use the original pretrained model of Bosselut et al., 2019 (please load the model from their repository). We recommend setting the flag.

If you run pipeline.py, it will generate a file with all generated paths in the -out-directory. 
In addition, at the beginning of running process a file named “forwardchaining_inputpairs_includeReverse [False / True] _ [input_filename] .tsv” is created in the forwardchaining/ directory, in which all possible relations after PoS filtering for each source-target-pairs are listed. 

Below we give an example command for running the code, with our default parameters and settings that we also used in our paper:

python pipeline.py -inputfile file.tsv -experiment_name "experiment1" -out paths/ -sim cos -emb numberbatch-en-17.06.txt -hops 3 -beams 10 -contextsimthresh 0.7 -keep all -threshold 0.95 -pos_filter -lemma_check -reverse

If you use the code, please cite: Becker, M., Korfhage, K., Paul, D., and Frank, A. (2021). CO-NNECT: A Framework for Revealing Commonsense Knowledge Paths as Explicitations of Implicit Knowledge in Texts. IWCS – International Conference on Computational Semantics.

Bosselut, A., Rashkin, H., Sap, M., Malaviya, C., Celikyilmaz, A., and Choi, Y. (2019). COMET: Commonsense transformers for automatic knowledge graph construction. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 4762–4779, Florence, Italy.




