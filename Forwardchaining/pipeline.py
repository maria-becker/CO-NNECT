
import os
import re
import sys
import csv
import pstats
import cProfile
import string
import pickle
import spacy
import subprocess
from glob import iglob
from shutil import copyfile
from datetime import datetime
from operator import itemgetter
from itertools import product
from argparse import ArgumentParser
from gensim.models import KeyedVectors
from cos_sim import cos_similarity
from subprocess import PIPE, STDOUT

class Sentence:

    def __init__(self, sent_id:str, sent:str, concepts:list):

        self.sent_id = sent_id
        self.sent = sent
        self.concepts = concepts


        
def get_pos(concept, nlp):
    # pos tag can be "n" (=noun), "v" (=verb) or "a" (=adjective)
    pos = [token.pos_ for token in nlp(concept) if token.dep_ == 'ROOT'][0]
    if pos in ['NOUN', 'PROPN', 'PRON', 'NUM']:
        return 'NP'
    elif pos in ['VERB', 'AUX']:
        return 'VP'
    elif pos == 'ADJ':
        return 'ADJP'
    else:
        #print("="*50, concept, pos)
        return '?'
        #sys.exit()
        


def is_valid_source_pos(source_pos, relation):
    # takes a (lowercased) relation and the pos tag ('n', 'v' or 'a') of a concept
    # returns true if the source concept in the relation may have this pos tag

    if relation in {'isa', 'usedfor', 'hasa', 'capableof', 'desires', 'createdby', 'partof', 'hasproperty', 'madeof', 'atlocation', 'definedas', 'symbolof', 'receivesaction', 'causesdesire',
                    'isareverse', 'hasareverse', 'partofreverse', 'madeofreverse', 'atlocationreverse', 'definedasreverse', 'symbolofreverse'}:
        return source_pos == 'NP'
    elif relation in {'hasprerequisite', 'motivatedbygoal', 'hassubevent', 'hasfirstsubevent', 'haslastsubevent',
                      'usedforreverse', 'capableofreverse', 'desiresreverse', 'createdbyreverse', 'receivesactionreverse', 'hasprerequisitereverse', 'motivatedbygoalreverse', 'causesdesirereverse'}:
        return source_pos == 'VP'
    elif relation in {'causes', 'causesreverse', 'hassubeventreverse', 'hasfirstsubeventreverse', 'haslastsubeventreverse'}:
        return (source_pos in {'NP', 'VP'})
    elif relation == 'haspropertyreverse':
        return source_pos == 'ADJP'
    else:
        # if relation is unknown
        print(relation)
        raise Exception


def is_valid_target_pos(target_pos, relation):
    # takes a (lowercased) relation and the pos tag ('n', 'v' or 'a') of a concept
    # returns true if the target concept in the relation may have this pos tag

    if relation in {'isa', 'hasa', 'partof', 'madeof', 'atlocation', 'definedas', 'symbolof',
                    'isareverse', 'usedforreverse', 'hasareverse', 'capableofreverse', 'desiresreverse', 'createdbyreverse', 'partofreverse', 'haspropertyreverse', 'madeofreverse', 'atlocationreverse', 'definedasreverse', 'symbolofreverse', 'receivesactionreverse', 'causesdesirereverse'}:
        return target_pos == 'NP'
    elif relation in {'usedfor', 'capableof', 'desires', 'createdby', 'receivesaction', 'hasprerequisite', 'motivatedbygoal', 'causesdesire',
                      'hasprerequisitereverse', 'motivatedbygoalreverse', 'hassubeventreverse', 'hasfirstsubeventreverse', 'haslastsubeventreverse'}:
        return target_pos == 'VP'
    elif relation == 'hasproperty':
        return target_pos == 'ADJP'
    elif relation in {'causes', 'hassubevent', 'hasfirstsubevent', 'haslastsubevent', 'causesreverse'}:
        return (target_pos in {'NP', 'VP'})
    else:
        # if relation is unknown
        print(relation)
        raise Exception

def valid_pos_tags(source_pos, target_pos, relation):
    # takes a tripel of two concepts' pos tags and a relation
    # returns true if the relation can take these pos tag, else false

    return (is_valid_source_pos(source_pos, relation) and is_valid_target_pos(target_pos, relation))

        
def upper_relation(relation):

    if relation == 'desires':
        return 'Desires'
    elif relation == 'motivatedbygoal':
        return 'MotivatedByGoal'
    elif relation == 'receivesaction':
        return 'ReceivesAction'
    elif relation == 'hasa':
        return 'HasA'
    elif relation == 'causes':
        return 'Causes'
    elif relation == 'hasprerequisite':
        return 'HasPrerequisite'
    elif relation == 'hassubevent':
        return 'HasSubevent'
    elif relation == 'hasproperty':
        return 'HasProperty'
    elif relation == 'capableof':
        return 'CapableOf'
    elif relation == 'usedfor':
        return 'UsedFor'
    elif relation == 'atlocation':
        return 'AtLocation'
    elif relation == 'isa':
        return 'IsA'
    if relation == 'desiresreverse':
        return 'DesiresReverse'
    elif relation == 'motivatedbygoalreverse':
        return 'MotivatedByGoalReverse'
    elif relation == 'receivesactionreverse':
        return 'ReceivesActionReverse'
    elif relation == 'hasareverse':
        return 'HasAReverse'
    elif relation == 'causesreverse':
        return 'CausesReverse'
    elif relation == 'hasprerequisitereverse':
        return 'HasPrerequisiteReverse'
    elif relation == 'hassubeventreverse':
        return 'HasSubeventReverse'
    elif relation == 'haspropertyreverse':
        return 'HasPropertyReverse'
    elif relation == 'capableofreverse':
        return 'CapableOfReverse'
    elif relation == 'usedforreverse':
        return 'UsedForReverse'
    elif relation == 'atlocationreverse':
        return 'AtLocationReverse'
    elif relation == 'isareverse':
        return 'IsAReverse'
    else:
        raise Exception('Unknown relation {}. Please fix.'.format(str(relation)))


def main():
    
    # command line arguments
    parser = ArgumentParser(description='generate concepts with COMET for IKAT dataset')
    parser.add_argument('-out',
                        required=True,
                        dest='outpath',
                        help='The output directory.')
    parser.add_argument('-start_idx',
                        required=False,
                        type=int,
                        default=0,
                        dest='start_idx',
                        help='start with Text at index given')
    parser.add_argument('-pos_filter',
                        required=False,
                        dest='pos_filter',
                        help='apply pos-rule to filter concepts according to relation type',
                        default=False,
                        action='store_true')
    parser.add_argument('-lemma_check',
                        required=False,
                        dest='lemma_check',
                        help='exclude pairs with identical lemmas',
                        default=False,
                        action='store_true')
    parser.add_argument('-hops',
                        required=False,
                        dest='hops',
                        help='How many hops to perform in ForwardChaining. Every hop is one COMET prediction.',
                        type=int,
                        default=1)
    parser.add_argument('-beams',
                        required=False,
                        default=2,
                        dest='beams',
                        type=int,
                        help='How many predictions to make for each entity-relation-pair')
    parser.add_argument('-threshold',
                        required=False,
                        default=0.8,
                        type=float,
                        dest='threshold',
                        help='Cutoff similarity threshold between predicted target and actual target')
    parser.add_argument('-sim',
                        required=False,
                        default='cos',
                        dest='sim',
                        help='What similarity measure to use for comparison of predicted and actual target')
    parser.add_argument('-contextsimthresh',
                        required=False,
                        default=0.5,
                        type=float,
                        dest='context_sim_thresh',
                        help='Cutoff similarity threshold between predicted target and target sentence')
    parser.add_argument('-keep',
                        required=False,
                        default='all',
                        dest='keep',
                        help='How many of the top context similarity predictions to keep for the next steps. Can be "all" or an int.')
    parser.add_argument('-emb',
                        required=True,
                        dest='emb',
                        help='Path to the embeddings file to be used for similarity calculation. Either GoogleNewsVectors or Numberbatch have been tested.')
    parser.add_argument('-reverse',
                        required=False,
                        default=False,
                        dest='reverse',
                        action='store_true',
                        help='Whether or not to include reverse relations.')
    parser.add_argument('-gold',
                        required=False,
                        default=False,
                        dest='gold',
                        action='store_true',
                        help='Whether or not to use gold (within) relations.')
    parser.add_argument('-inputfile',
                        required=True,
                        dest='inputfile',
                        help='Path and filename of inputfile which contains the entity selection to be checked')
    parser.add_argument('-experiment_name',
                        required=True,
                        default='default_experiment',
                        dest='experiment_name',
                        help='Name of the experiment, should be a unique handle to avoid cross-overwriting of temp files.')

    args = parser.parse_args()

    if args.sim not in ['cos', 'wmd']:
        raise NotImplementedError(f'"{args.sim}" is not implemented as a similarity metric.')

    if args.keep != 'all':
        try:
            args.keep = int(args.keep)
        except:
            print(type(args.keep))
            raise ValueError('"-keep" must be either an int or "all" if specified. Default is all.')

    # create output directory if not exists
    if not os.path.exists(args.outpath):
            os.mkdir(args.outpath)

    # path where corpus data is saved to and loaded from, should be unique for every experiment running in parallel to avoid cross-overwriting
    datapath = "parallel_data/" + args.experiment_name + "/"
            
            
    print(datetime.now(), 'args parsed')
    
    if args.reverse:
        relations=["Desires", "MotivatedByGoal", "ReceivesAction", "HasA",
                   "Causes", "HasPrerequisite", "HasSubevent", "HasProperty",
                   "CapableOf", "UsedFor", "AtLocation", "IsA", "DesiresReverse", "MotivatedByGoalReverse", "ReceivesActionReverse", "HasAReverse",
                   "CausesReverse", "HasPrerequisiteReverse", "HasSubeventReverse", "HasPropertyReverse",
                   "CapableOfReverse", "UsedForReverse", "AtLocationReverse", "IsAReverse"]
        model_name = '1e-05_adam_64_33500.pickle'
        pred_dict_filename = "predictions_dict_reverse.pickle"
        dataloader_path = "comet_scripts_reverse/data/make_conceptnet_data_loader.py"
        generator_path = "comet_scripts_reverse/generate/generate_conceptnet_beam_search.py"
        if not os.path.exists(datapath):
            os.makedirs(datapath)
            for fn in iglob('data_reverse/conceptnet/*.txt'):
                copyfile(fn, datapath+fn.split("/")[-1])
    else:
        relations=["Desires", "MotivatedByGoal", "ReceivesAction", "HasA",
               "Causes", "HasPrerequisite", "HasSubevent", "HasProperty",
               "CapableOf", "UsedFor", "AtLocation", "IsA"]
        model_name = 'conceptnet_pretrained_model.pickle'
        pred_dict_filename = "predictions_dict_noreverse.pickle"
        dataloader_path = "comet_scripts_noreverse/data/make_conceptnet_data_loader.py"
        generator_path = "comet_scripts_noreverse/generate/generate_conceptnet_beam_search.py"
        if not os.path.exists(datapath):
            os.makedirs(datapath)
            for fn in iglob('data_noreverse/conceptnet/*.txt'):
                copyfile(fn, datapath+fn.split("/")[-1])
        
    generations_path = model_name.replace(".pickle", "/" + args.experiment_name + "/")
    if not os.path.exists(generations_path):
        os.makedirs(generations_path)
    
    loaded_data_save_path = datapath + "processed/generation"
    
    input_filename = args.inputfile.split("/")[-1].split(".")[0]

    
    # get adjacency and argumentative relation info from ikat to include in output later
    # dict where keys are tuples of text_id and sent ids and values are tuples of adjacency info and arg rel type
    # not needed for additional corpora, thus commented
    """
    ikat_pairs_metadata = dict()
    with open("../entity_extraction/IKAT_v003.tsv") as ikat:
        for row in ikat:
            if re.match("Text:", row):
                current_text_id = row.split("\t")[0].strip().replace("Text:  micro_","")
            elif re.match("e\d\d?-e\d\d?", row):
                cells = row.strip("\n").split("\t")
                sent1_id, sent2_id = cells[0].split("-")
                ikat_pairs_metadata[(current_text_id, sent1_id, sent2_id)] = (cells[1], cells[2])
                ikat_pairs_metadata[(current_text_id, sent2_id, sent1_id)] = (cells[1], cells[2])
    """
    
    # get previously extracted entitites as input for comet
    with open(args.inputfile) as f:
        texts = dict()
        for line in f:
            try:
                text_id, sent_id, sent, concepts = line.strip("\n").split("\t")
            except:
                print("#", line, "#")
                raise Exception
            if text_id not in texts:
                texts[text_id] = dict()
            texts[text_id][sent_id] = Sentence(sent_id, sent, [(concept.split("|")[0], concept.split("|")[1]) for concept in concepts.strip("[").strip("]").split("][")])
            
    print(datetime.now(), 'texts imported')

    nlp = spacy.load('en')
    
    # save all predictions that have already been made in here, to save computation time
    # get backuped prediction dict from previous runs from .pickle if one exists - this way, we don't have to re-predict things that were already predicted in an earlier run
    try:
        with open(pred_dict_filename, "rb") as f:
            pred_dict = pickle.load(f)
    except:
        pred_dict = dict()

    # if all inputs have been cached before, the comet subprocesses shouldn't be executed because we can get all info from cache already
    # will be set to true if unseen inputs are found while writing comet input file
    unseen_inputs_exist = False
        
    # format comet inputdata properly
    # to be able to match the comet generations back with the sentence
    # write comet test file with our entities
    with open(datapath + 'test.txt', 'w') as comet_input:

        combinations = product(list({node for text in list(texts.values())[:] for sent in text.values() for node, pos in sent.concepts}), relations)
        for combi in combinations:
            concept, relation = combi
            if "|".join([concept, relation.lower()]) not in pred_dict:
                unseen_inputs_exist = True
                comet_input.write(f'{relation}\t{concept}\t{concept}\t1\n')
                

    print(datetime.now(), 'inputfile written for hop 1') 

    found_paths = list()

    list1 = list()
    list2 = list()

    # load lemmatizer that takes pos tags into account
    #lemmatizer = nlp.vocab.morphology.lemmatizer
    # cache lemmas as produced by spacy's nlp, so we don't have to double_check ones that we have already checked
    lemmas_dict = dict()

    #additionally write list1 to file
    with open("forwardchaining_inputpairs_includeReverse{}_{}.tsv".format(args.reverse, input_filename.replace("_", "-")), "w") as f:
        f.write("#text_id\t#sent1_id\t#sent2_id\t#adjacent\t#argrel\t#source\t#target\t#relations_incl_reverse\t#relations_without_reverse\n")

        for text_id, sents in list(texts.items())[:]:

            # this decides if pairs are created across or within sentences
            if args.gold:
                pairs = [(sent1.sent_id, sent2.sent_id) for sent1, sent2 in product(list(sents.values()), repeat=2) if (sent1.sent_id == sent2.sent_id)]
            else:
                pairs = [(sent1.sent_id, sent2.sent_id) for sent1, sent2 in product(list(sents.values()), repeat=2) if (sent1.sent_id != sent2.sent_id)]

            for sent1_id, sent2_id in pairs:
                for source, source_pos in sorted(list(set(sents[sent1_id].concepts))):
                    for target, target_pos in sorted(list(set(sents[sent2_id].concepts))):
                        # get lemmas of source and target to compare against each other, so we don't get paths between inflected forms (e.g. "produce -> produces")
                        if source in lemmas_dict:
                            source_lemmas = lemmas_dict[source]
                        else:
                            source_lemmas = " ".join([token.lemma_ for token in nlp(source)])
                            lemmas_dict[source] = source_lemmas
                        if target in lemmas_dict:
                            target_lemmas = lemmas_dict[target]
                        else:
                            target_lemmas = " ".join([token.lemma_ for token in nlp(target)])
                            lemmas_dict[target] = target_lemmas
                        if (source != target) and ((not args.lemma_check) or (source_lemmas != target_lemmas)):
                            adjacent = "N/A"
                            argrel = "N/A"
                            rels_with_reverse = list()
                            rels_wo_reverse = list()
                            for relation in relations:
                                if (not args.pos_filter) or (valid_pos_tags(source_pos, target_pos, relation.lower())):
                                    list1.append([text_id, sent1_id, sent2_id, target, source, 0, relation.lower()])
                                    rels_with_reverse.append(relation)
                                    if not ("Reverse" in relation):
                                        rels_wo_reverse.append(relation)
                            f.write(f"{text_id}\t{sent1_id}\t{sent2_id}\t{adjacent}\t{argrel}\t{source}\t{target}\t{'|'.join(rels_with_reverse)}\t{'|'.join(rels_wo_reverse)}\n") 

                                
    print(datetime.now(), 'predictions list prepared') 
    
    if (args.sim == 'cos') or (args.sim == 'wmd'):

        if args.emb.endswith(".bin"):
            binary = True
        elif args.emb.endswith(".txt"):
            binary = False
        else:
            raise Exception("The embeddings file name must either end in .bin for a binary file or .txt for a non-binary file.")
        try:
            model = KeyedVectors.load_word2vec_format(args.emb, binary=binary)
        except:
            raise Exception("The embeddings file you specified does not seem to exist! Please double-check the path and give me an embeddings file that actually exists. Thanks!")
        
        with open('stopwords.txt', 'r') as f:
            stops = set(f.read().splitlines())
        stops = stops.union(string.punctuation)
    

    print(datetime.now(), 'word2vec model loaded (for cos sim or wmd)') 
        
    # cache pos tagging of predictions that have already been tagged, to save computation time
    pred_pos_dict = dict()

    # save all similarities between a prediction and target that have already been calculated
    # this is also used to check for the "context" similarity, where we pick the beams with the highest similarity for further processing 
    sim_dict = dict()

    # collect not recognized inputs
    error_inputs = set()

    # dict to save which entity pairs have already formed a path, to create only the shortest possible paths
    # format: {microtext_id1 : {"unit1|unit2" : {"source_entity1|target_entity1", "source_entity1|target_entity2", "source_entity2|target_entity1", ...}, "unit1|unit3" : ...}, microtext_id2 : ...}
    # unit1|unit2 are sorted in ascending order of ids, source_entity and target_entity are sorted alphabetically, to cover both directionsa
    found_entity_pairs_dict = dict()
    
    print('DATA BEFORE FIRST HOP:')
    print('len found_paths:', len(found_paths))
    print('len list1:', len(list1))
    print('len list2:', len(list2))
    #print('len relations:', len(relations))
    print('len pred_dict:', len(pred_dict))
    print('len sim_dict:', len(sim_dict))
    
    print(datetime.now(), 'starting hop iteration') 
    
    # iterate hops
    for hop in range(args.hops):
        print(datetime.now(), "Hop {}".format(hop+1))

        print('DATA AT START OF HOP:')
        print('len found_paths:', len(found_paths))
        print('len list1:', len(list1))
        print('len list2:', len(list2))
        print('len relations:', len(relations))
        print('len pred_dict:', len(pred_dict)) 
        print('len sim_dict:', len(sim_dict))

        if unseen_inputs_exist:
        
            print(datetime.now(), "Loading data - starting first subprocess")
            # first run dataloader
            try:
                process_load = subprocess.run(
                    ['python', dataloader_path,
                     '--data_path', datapath[:-1],
                     '--save_path', loaded_data_save_path],
                    check=True, universal_newlines=True, stdout=PIPE, stderr=PIPE)
                #stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                #check=True, '''stdout=subprocess.PIPE,''' universal_newlines=True)
            except subprocess.CalledProcessError as e:
                #print(e.output)
                print(e.stdout)
                print(e.stderr)
                sys.stdout.flush()
                raise Exception
                #print(process_load.stderr)
                #print(process_load.stdout)
                #raise Exception
                
            print(datetime.now(), 'first subprocess (data loader) finished') 
            print(datetime.now(), "Generating predictions - starting second subprocess")
            # now comet generation itself
            try:
                process_generate = subprocess.run(
                    ['python', generator_path,
                     '--split', 'test',
                     '--beam', str(args.beams),
                     #'--model_name', '1e-05_adam_64_33500.pickle'],
                     '--model_name', model_name,
                     '--inputpath', loaded_data_save_path,
                     '--outputpath', generations_path[:-1]],
                    check=False, universal_newlines=True, stdout=PIPE, stderr=PIPE)
                #stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                #check=True, stdout=subprocess.PIPE, universal_newlines=True)
                #print(process_generate.stderr)
                #print(process_generate.stdout)
            except subprocess.CalledProcessError as e:
                print(e.stdout)
                print(e.stderr)
                raise Exception
                
            print(datetime.now(), 'second subprocess (generation) finished') 

            print(process_generate.stdout)
            print(process_generate.stderr)
            
            # get comet-predictions from file where they are saved
            with open(generations_path + "test.pickle", 'rb') as comet_prediction_file:
                predictions = pickle.load(comet_prediction_file)

            print(datetime.now(), 'predictions loaded into main program') 

            bla = True
            for pred in predictions:
                # add to global prediction dict shared over all hops
                if bla:
                    print(pred)
                    print("HANDLE", '|'.join([pred['e1'].strip(), pred['r'].replace(' ', '').replace('prequisite', 'prerequisite')]))
                    bla = False
                    pred_dict['|'.join([pred['e1'].strip(), pred['r'].replace(' ', '').replace('prequisite', 'prerequisite')])] = pred['beams']

            print(datetime.now(), 'predictions added to global prediction dict') 

            print('DATA AFTER LOADING COMET OUTPUT:')
            print('len found_paths:', len(found_paths))
            print('len list1:', len(list1))
            print('len list2:', len(list2))
            print('len relations:', len(relations))
            print('len pred_dict:', len(pred_dict))
            print('len sim_dict:', len(sim_dict)) 
            print('len predictions:', len(predictions))

        else:
            print(datetime.now(), 'no unseen inputs for comet to process - moving along with previously-cached predictions')        
                    
        # get predictions for each source entity from prediction dict
        for elm in list1:
            pred_dict_handle = '|'.join([elm[-3], elm[-1]])
            # to save information about paths already found for entity pairs later, so to only get the shortest paths
            unit_ids_handle = "|".join(sorted([elm[1], elm[2]]))
            entities_handle = "|".join(sorted([elm[3], elm[4]]))
            
            context_sims_of_beam = list()
            for beam_nr in range(args.beams):
                ####################################
                # only pick the ones with the highest similarity to target from the beam here
                ####################################
                try:
                    pred = pred_dict[pred_dict_handle][beam_nr]
                except Exception as e:
                    try:
                        pred_dict_handle = '|'.join([elm[-3].replace("-", " - ").strip(), elm[-1]])
                        pred = pred_dict[pred_dict_handle][beam_nr]
                    except:
                        # UNRESOLVED ERROR
                        # these cases should still be handled better in the future!
                        continue
                    continue
                # get the "context" similarity between prediction and target
                sim_dict_handle = '|'.join([elm[3], pred])
                if sim_dict_handle in sim_dict:
                    sim_score = sim_dict[sim_dict_handle]
                else:
                    if args.sim == 'cos':
                        sim_score = cos_similarity(elm[3].split(" "), pred.split(" "), model, stops)
                    elif args.sim == 'wmd':
                        # wm similarity is negative wmd
                        sim_score = 1/(1+model.wmdistance(elm[3].split(' '), pred.split(' ')))
                        if sim_score == float('inf'):
                            sim_score = -1 # to assure these won't get over any thresholds
                    sim_dict[sim_dict_handle] = sim_score
                context_sims_of_beam.append((pred, sim_score))
            # sort by highest context similarity
            context_sims_of_beam.sort(key=lambda x: x[1], reverse=True)
            # determine the maximum number of items n from the beam to keep if their context sim is above the threshold
            if args.keep == 'all':
                max_nr_of_items_to_keep = len(context_sims_of_beam)
            else:
                max_nr_of_items_to_keep = args.keep
            # check for each item (up to n, as determined above) if it is above the context sim threshold
            for pred, sim_score in context_sims_of_beam[:max_nr_of_items_to_keep-1]:
                if sim_score >= args.context_sim_thresh:
                    # if path breakoff threshold similarity is also reached, save as complete path
                    if args.pos_filter:
                        if pred in pred_pos_dict:
                            pred_pos = pred_pos_dict[pred]
                        else:
                            pred_pos = get_pos(pred, nlp)
                            pred_pos_dict[pred] = pred_pos
                        if is_valid_target_pos(pred_pos, elm[-1]):
                            valid_target_pos = True
                        else:
                            valid_target_pos = False
                    else:
                        valid_target_pos = True
                    if valid_target_pos:
                        if sim_score >= args.threshold:
                            new_elm = elm + [pred, sim_score]
                            found_paths.append(new_elm)
                            # save info that a path has been found for this entity pair, so we don't need to keep looking for further paths in the next hop
                            if elm[0] not in found_entity_pairs_dict:
                                found_entity_pairs_dict[elm[0]] = dict()
                            if unit_ids_handle not in found_entity_pairs_dict[elm[0]]:
                                found_entity_pairs_dict[elm[0]][unit_ids_handle] = set()
                            found_entity_pairs_dict[elm[0]][unit_ids_handle].add(entities_handle)
                        # if path breakoff threshold similarity is not yet reached, keep incomplete path as input for the next hop
                        else:
                            # check if a path has already been found for this pair of entities
                            # if yes, do not keep it for the next hop because we only want the shortest paths for each pair
                            # if no, keep for next hop as usual
                            for relation in relations:
                                # POS FILTERING
                                # check if prediction is a valid input to each new relation
                                if args.pos_filter:
                                    if pred in pred_pos_dict:
                                        source_pos = pred_pos_dict[pred]
                                    else:
                                        source_pos = get_pos(pred, nlp)
                                        pred_pos_dict[pred] = source_pos
                                    if is_valid_source_pos(source_pos, relation.lower()):
                                        new_elm = elm + [pred, sim_score, relation.lower()]
                                        list2.append(new_elm)
                                else:
                                    new_elm = elm + [pred, sim_score, relation.lower()]
                                    list2.append(new_elm)
                # break if elm is below context sim threshold, because all consecutive elms in this beam will then also be below
                else:
                    break
                
        print(datetime.now(), 'all predictions matched with corresponding sentence pairs')
        print(datetime.now(), 'all similarities calculated and all paths sorted')

        # update list1 with new incomplete paths and clear list2 (temporary storage only)
        # check if a path has already been found for this pair of entities
        # if yes, do not keep it for the next hop because we only want the shortest paths for each pair
        # if no, keep for next hop as usual
        path_not_shortest_for_entity_pair_counter = 0
        list1 = list()
        for elm in list2:
            if not ((elm[0] in found_entity_pairs_dict) and ("|".join(sorted([elm[1], elm[2]])) in found_entity_pairs_dict[elm[0]]) and ("|".join(sorted([elm[3], elm[4]])) in found_entity_pairs_dict[elm[0]]["|".join(sorted([elm[1], elm[2]]))])):
                list1.append(elm)
            else:
                path_not_shortest_for_entity_pair_counter += 1

        print('DATA AFTER CALCULATING SIMILARITIES AND ADDING NEW PATHS TO FOUND_PATHS OR LIST1, RESPECTIVELY:')
        print('len found_paths:', len(found_paths))
        print('len list1:', len(list1))
        print('len list2:', len(list2))
        print('len relations:', len(relations))
        print('len pred_dict:', len(pred_dict))
        print('len sim_dict:', len(sim_dict))
        print('nr of paths sorted out bc there were shorter ones found for entity pair before:', path_not_shortest_for_entity_pair_counter)

        list2 = list()
        print(datetime.now(), 'writing a new inputfile for comet')
        # write a new inputfile for comet for the next hop
        with open(datapath + 'test.txt', 'w') as comet_input:
            combinations = {(elm[-3].replace("-", " - "), elm[-1]) for elm in list1}
            print(datetime.now(), '###### TOTAL INCOMPLETE PATHS: ', len(combinations))
            unseen_inputs = 0
            for concept, relation in combinations:
                # only calculate new predictions if the input has not been processed in a previous hop already
                # if it has, it is already in pred_dict and predictions can be obtained from there
                if '|'.join([concept, relation]) not in pred_dict:
                    unseen_inputs += 1
                    formatted_relation = upper_relation(relation)
                    comet_input.write(f'{formatted_relation}\t{concept}\t{concept}\t1\n')
            print(datetime.now(), '###### UNSEEN INPUTS: ', unseen_inputs)
            if unseen_inputs == 0:
                unseen_inputs_exist = False
            else:
                unseen_inputs_exist = True

        print(datetime.now(), 'new inputfile written')

        # write everything that was found up until this hop to outputfile
        if "GoogleNews" in args.emb:
            embeddings_name = "GoogleNews"
        elif "numberbatch" in args.emb:
            embeddings_name = "Numberbatch"
        else:
            embeddings_name = args.emb.split("/")[-1].split(".")[-2]
        with open(args.outpath + '/{}_reverse{}_{}_{}_contextsimthresh{}_keep{}_pathbreakthresh{}_posfilter{}_lemmacheck{}_{}hops.tsv'.format(input_filename.replace("_", "-"), args.reverse, args.sim, embeddings_name, args.context_sim_thresh, args.keep, args.threshold, args.pos_filter, args.lemma_check, hop+1), 'w') as f:
            f.write('#Microtext_ID\t#Pair_Unit1\t#Pair_Unit2\t#Adjacent?\tArgumentative_Relation\t#Target\t#Source')
            [f.write('\t#Relation{}\t#Predicted_Target{}\t#Similarity_To_Target{}'.format(hop_idx+1, hop_idx+1, hop_idx+1)) for hop_idx in range(hop+1)]
            f.write('\n')
            for path in sorted(found_paths, key = lambda x : (x[0], x[1], x[2], len(x), x[3], x[4])):
                adjacency = "N/A"
                arg_rel = "N/A"
                f.write(f'{path[0]}\t{path[1]}\t{path[2]}\t{adjacency}\t{arg_rel}\t{path[3]}\t{path[4]}')
                [f.write(f'\t{elm}') for elm in path[6:]]
                f.write('\n')
        print(datetime.now(), 'results from this hop written to file')
        print(datetime.now(), 'starting next hop if any are left')

        print('DATA AT THE END OF HOP:')
        print('len found_paths:', len(found_paths))
        print('len list1:', len(list1))
        print('len list2:', len(list2))
        print('len relations:', len(relations))
        print('len pred_dict:', len(pred_dict))
        print('len sim_dict:', len(sim_dict))

    print(datetime.now(), 'hops completed')
    print(datetime.now(), 'saving predictions dict for the future')

    with open(pred_dict_filename, "wb") as f:
        pickle.dump(pred_dict, f)
    
    print(datetime.now(), 'predictions dict saved')    
    print(datetime.now(), 'all done')
        
    return None


if __name__ == '__main__':

    main()
