###############################
README - Word-Vector Model
###############################
1. Open terminal

2. Stand in the directory: 'med-ner/word-vector-model'

3. Run 'python model.py --seed_term_lists_path <PATH> --corpus_path <PATH> --output_path <PATH> --input_path <PATH>'

Replace <PATH> for each provided arguments with its path.

4. Wait for the program to run. When it is done, the last printout in the terminal should be 'Done Classifying Entities'

5. In the output path, a file named classified_documents.txt is generated. The file contains the input document with classifications based on the categories.

The required arguments are:
	--seed_term_list_path is the path to directory where all seed term list are, one seed term list for each textfile, separated by new line 
	--corpus_path is the path to corpus that will be used to generate category vectors and vocabulary
	--output_path is the path to directory where output and generated files will be
	--input_path is the path to input file that should be classified

The optional arguments are:
	--with_weighted_context sets whether or not the context words are weighted by distance
	--internal_weight is the weight of internal words, used in relation to context weights
	--context_weight is the weight of context words, used in relation to internal weights
	--frequency_threshold is the number of times a word must occur in the corpus in order to be put in the vocabulary
	--classification_threshold determines how much greater the similarity between the category that the entity is most similar to must be in comparison to the other categories
	--similarity_threshold determines how similar the vectors must be in order to be classified at all
	--idf_threshold determines how great the IDF value must be for a given entity candidate in order to still be considered as a candidate
	--context_window is the number of words on each side of the internal words that will be be considered as context words
	--clean_directory removes all the files in the output directory if there are any
    --nbr_of_threads is the number of threads to use when classifying entities, where 0 is the maximum number of cores available (be aware of the overhead when running multi-threaded)
