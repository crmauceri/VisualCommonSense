import os, sys, json, time
from multiprocessing import Process, Queue, JoinableQueue, Lock, cpu_count, freeze_support
from threading import Thread
import collections
import numpy as np
import scipy.sparse as sparse
import scipy.io as spio
import itertools as it
import re
import cProfile
import psutil
import shutil

#Module-level function for pickling nested defaultdicts
def defaultdict_set():
    return collections.defaultdict(set)

# ######################################
# Load File Process
# ######################################
class GetDatasetProcess(Process):
    def __init__(self, filepath_q, init_dict, do_pairs, has_vocabulary,
                 max_memory_percent, total_num_processes, out_dir, save_description):
        Process.__init__(self)

        #Inputs for both methods
        self.filepath_q = filepath_q
        self.max_memory = max_memory_percent
        self.total_num_processes = total_num_processes
        self.total_dict = 0
        self.current_dict = 0
        self.filenum = 0
        self.out_dir = out_dir
        self.save_description = save_description
        self.pattern = re.compile(r'[^\w:=?]', flags=0)

        # Method specific variables
        if has_vocabulary:
            #Inputs
            self.concept_vocabulary = list(init_dict['concept_vocabulary'])
            self.vocab_index = init_dict['vocab_index'].copy()

            self.photoID = set([])
            self.wordlist = list(init_dict['wordlist'])
            self.wordlist_index = init_dict['wordlist_index'].copy()

            self.num_words = init_dict['num_words']
            self.num_vocab = init_dict['num_vocab']

            self.vocab_feature = init_dict['vocab_feature'].copy()
            self.synonym_feature = init_dict['synonym_feature'].copy()
            self.num_words_in_vocab = init_dict['num_words_in_vocab'].copy()

            if do_pairs:
                self.query_vocabulary = list(init_dict['query_vocabulary'])
                self.query_index = init_dict['query_index'].copy()
                self.num_query = init_dict['num_query']

    def concept_vocab_feature(self, photo_dict):
        num_photos = len(photo_dict)
        batch = sparse.lil_matrix((num_photos, self.num_words))
        batch_keys = {}
        batch_ind = 0
        for photokey in photo_dict:
            all_concepts = [re.sub(self.pattern, '', concept) for concept in photo_dict[photokey]['concepts']]
            all_concepts = [concept for concept in all_concepts if len(concept) > 0]
            concept_indices = list(set([self.wordlist_index[c] for c in all_concepts if c in self.wordlist]))
            if len(concept_indices) > 0:
                batch[batch_ind, concept_indices] = 1
                batch_keys[photokey] = batch_ind
                batch_ind += 1

        batch = sparse.csr_matrix(batch)
        batch_patterns = batch.dot(np.transpose(self.vocab_feature)) >= 1
        batch_patterns = batch_patterns.dot(np.transpose(self.synonym_feature)) >= 1

        return batch_patterns, batch_keys

    def get_next_photo_dict(self):
        filepath = self.filepath_q.get()

        if filepath == "Stop":
            # Poison pill shutdown
            print "Exiting " + self.name
            self.filepath_q.task_done()
            return {}, True
        else:
            try:
                with open(filepath, 'r') as fp:
                    file_dict = json.load(fp)
            except:
                print "Unexpected error on {}: {}".format(filepath, sys.exc_info()[0])
            else:
                processed_photo_list = file_dict['processed_photo_list']
                photo_query_list = file_dict['photo_query_list']
                tag_list = file_dict['tag_list']
                tag_image_list = file_dict['tag_image_list']

                processed_photo_dict = {photo['url']: {'owner': photo['owner']} for photo in processed_photo_list}

                photo_query_dict = collections.defaultdict(list)
                for pair in photo_query_list:
                    photo_query_dict[pair['image_key']].append(pair['query_str'])

                tag_image_dict = collections.defaultdict(list)
                for pair in tag_image_list:
                    tag_image_dict[pair['image_key']].append(pair['concept_key'])

                processed_photo_dict = {
                    k: dict(processed_photo_dict.get(k, {}), **{'queries': photo_query_dict.get(k, {})}) for k in
                    photo_query_dict.keys()}
                processed_photo_dict = {
                    k: dict(processed_photo_dict.get(k, {}), **{'concepts': tag_image_dict.get(k, {})}) for k in
                    tag_image_dict.keys()}
                return processed_photo_dict, False
            
            return {}, False

    def save_to_file(self, out_dict):
        print "Saving {} dicts of total {} dicts".format(self.current_dict, self.total_dict)

        save_path = os.path.join(self.out_dir, '{0}_{1:05d}_temp_out_dict.txt'.format(self.name, self.filenum))
        while os.path.exists(save_path):
            self.filenum += 1
            save_path = os.path.join(self.out_dir, '{0}_{1:05d}_temp_out_dict.txt'.format(self.name, self.filenum))

        with open(save_path, 'w') as fp:
            json.dump(out_dict, fp)

        self.filenum += 1

    def at_max_memory(self):
        p = psutil.Process(self.pid)
        mem = p.memory_percent()
        return mem > (self.max_memory/self.total_num_processes)

    def run(self):
        print "Starting " + self.name
        while True:
            if self.at_max_memory():
                out_dict = self.get_out_dict()
                self.save_to_file(out_dict)
                self.clear_variables()

            photo_dict, break_condition = self.get_next_photo_dict()
            if break_condition:
                self.save_to_file(self.get_out_dict())
                self.clear_variables()
                break
            else:
                self.process_dict(photo_dict)
                self.total_dict += 1
                self.current_dict += 1

    def process_dict(self):
        raise NotImplementedError()

    def get_out_dict(self):
        raise NotImplementedError()

    def clear_variables(self):
        raise NotImplementedError()


class FindVocabularyProcess(GetDatasetProcess):
    def __init__(self, filepath_q, out_q, init_dict, max_memory_percent, total_num_processes, out_dir, save_description):
        GetDatasetProcess.__init__(self, filepath_q, init_dict, False, False,
                                   max_memory_percent, total_num_processes, out_dir, save_description)

        self.photoID = set([])
        self.ownerID = set([])
        self.concepts_per_image = []
        self.concept_owners = collections.defaultdict(set)
        self.concept_images = collections.defaultdict(set)
        self.query_owners = collections.defaultdict(set)
        self.query_images = collections.defaultdict(set)
        self.pattern = re.compile(r'[^\w:=?]', flags=0)

        #Final output
        self.out_q = out_q

    def process_dict(self, photo_dict):
        for photokey in photo_dict:
            concepts = [re.sub(self.pattern, '', concept) for concept in photo_dict[photokey]['concepts']]
            concepts = [concept for concept in concepts if len(concept) > 0]
            owner = photo_dict[photokey]['owner']
            self.ownerID.add(owner)
            self.photoID.add(photokey)
            self.concepts_per_image.append(len(concepts))

            for concept in concepts:
                self.concept_owners[concept].add(owner)
                self.concept_images[concept].add(photokey)

            queries = photo_dict[photokey]['queries']
            for query in queries:
                self.query_owners[query].add(owner)
                self.query_images[query].add(photokey)

    def get_out_dict(self):
        self.photoID = list(self.photoID)
        self.ownerID = list(self.ownerID)

        for key in self.concept_owners:
            self.concept_owners[key] = list(self.concept_owners[key])
            self.concept_images[key] = list(self.concept_images[key])

        for key in self.query_owners:
            self.query_owners[key] = list(self.query_owners[key])
            self.query_images[key] = list(self.query_images[key])

        return {'photoID': self.photoID,
                    'ownerID': self.ownerID,
                    'concepts_per_image': self.concepts_per_image,
                    'concept_owners': self.concept_owners,
                    'concept_images': self.concept_images,
                    'query_owners': self.query_owners,
                    'query_images': self.query_images}

    def clear_variables(self):
        self.photoID[:] = []
        self.photoID = set(self.photoID)
        self.ownerID[:] = []
        self.ownerID = set(self.ownerID)
        self.concepts_per_image[:] = []
        self.concept_owners.clear()
        self.concept_images.clear()
        self.query_owners.clear()
        self.query_images.clear()


class PairwiseOccurrenceProcess(GetDatasetProcess):
    def __init__(self, filepath_q, init_dict, out_dir, save_description, total_num_processes, max_memory_percent, lock):
        GetDatasetProcess.__init__(self, filepath_q, init_dict, True, True,
                                   max_memory_percent, total_num_processes, out_dir, save_description)

        self.lock = lock

        self.query_vocabulary = list(init_dict['query_vocabulary'])
        self.query_index = init_dict['query_index'].copy()
        self.num_query = init_dict['num_query']

        self.owners_set = set([]) #all unique owners
        self.out_dict = collections.defaultdict(self.out_dict_element)
        self.query_frequency = collections.defaultdict(set)

    def out_dict_element(self):
        return {'codict_owner': collections.defaultdict(set),
            'codict_images': collections.defaultdict(set),
            'codict_query': collections.defaultdict(set),
            'vocab_frequency': set([]),
            'vocab_images': set([])}

    def process_dict(self, photo_dict):
        batch_patterns, batch_keys = self.concept_vocab_feature(photo_dict)
        for photokey in photo_dict:
            photo = photo_dict[photokey]

            if photokey in batch_keys:
                pattern = batch_patterns[batch_keys[photokey], :]

                owner = photo["owner"]
                self.owners_set.add(owner)

                query_indices = [self.query_index[query] for query in photo['queries']]
                for ind in query_indices:
                    self.query_frequency[ind].add(owner)

                pattern_indices = sorted(sparse.find(pattern)[1])
                for p in pattern_indices:
                    self.out_dict[p]["vocab_frequency"].add(owner)
                    self.out_dict[p]["vocab_images"].add(photokey)

                for combo in it.combinations(pattern_indices, 2):
                    self.out_dict[combo[1]]["codict_owner"][combo[0]].add(owner)
                    self.out_dict[combo[1]]["codict_images"][combo[0]].add(photokey)
                    self.out_dict[combo[1]]["codict_query"][combo[0]].update(query_indices)

    def get_out_dict(self):
        for vocab in self.out_dict:
            self.out_dict[vocab]['vocab_frequency'] = list(self.out_dict[vocab]['vocab_frequency'])
            self.out_dict[vocab]['vocab_images'] = list(self.out_dict[vocab]['vocab_images'])

            for v in self.out_dict[vocab]['codict_owner']:
                self.out_dict[vocab]['codict_owner'][v] = list(self.out_dict[vocab]['codict_owner'][v])
                self.out_dict[vocab]['codict_images'][v] = list(self.out_dict[vocab]['codict_images'][v])
                self.out_dict[vocab]['codict_query'][v] = list(self.out_dict[vocab]['codict_query'][v])

        for ind in self.query_frequency:
            self.query_frequency[ind] = list(self.query_frequency[ind])

        self.owners_set = list(self.owners_set)

        return self.out_dict

    def save_to_file(self, out_dict):
        print "Saving {} dicts of total {} dicts".format(self.current_dict, self.total_dict)

        save_dir = os.path.join(self.out_dir, 'owners_set')
        self.lock.acquire()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.lock.release()

        save_path = os.path.join(save_dir, '{0}.txt'.format(self.name))
        with open(save_path, 'a') as fp:
            fp.write("\n".join(self.owners_set))

        save_dir = os.path.join(self.out_dir, 'query_frequency')
        self.lock.acquire()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.lock.release()

        save_path = os.path.join(save_dir, '{0}_{1:05d}.txt'.format(self.name, self.filenum))
        with open(save_path, 'w') as fp:
            json.dump(self.query_frequency, fp)

        #For out_dict, I want to split the dictionary by each vocabulary used
        for vocab in self.out_dict:
            save_dir = os.path.join(self.out_dir, 'by_vocab', "{}".format(vocab))
            self.lock.acquire()
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            self.lock.release()

            save_path = os.path.join(save_dir, '{0}_{1:05d}.txt'.format(self.name, self.filenum))
            with open(save_path, 'w') as fp:
                json.dump(self.out_dict[vocab], fp)

        self.filenum += 1

    def clear_variables(self):
        self.out_dict.clear()
        self.owners_set[:] = [] #all unique owners
        self.owners_set = set(self.owners_set)
        self.query_frequency.clear()
        self.current_dict = 0


class ApproxPairwiseOccurrenceProcess(GetDatasetProcess):
    def __init__(self, filepath_q, out_q, init_dict, max_memory_percent, total_num_processes, out_dir, save_description):
        GetDatasetProcess.__init__(self, filepath_q, init_dict, False, True,
                                   max_memory_percent, total_num_processes, out_dir, save_description)
        self.image_count = 0
        self.comatrix = sparse.csr_matrix((self.num_vocab, self.num_vocab))
        self.vocab_frequency_vector = np.zeros((self.num_vocab, 1))

        #Final output
        self.out_q = out_q

    def approximate_pairwise_occurrence(self, photo_dict):
        batch_patterns, batch_keys = self.concept_vocab_feature(photo_dict)
        batch_patterns = batch_patterns.astype(float).transpose()
        self.image_count += len(batch_keys)
        self.comatrix = self.comatrix + batch_patterns.dot(batch_patterns.transpose())
        self.vocab_frequency_vector = self.vocab_frequency_vector + batch_patterns.sum(axis=1)

    def run(self):
        print "Starting " + self.name
        while True:
            photo_dict, break_condition = self.get_next_photo_dict()
            if break_condition:
                break
            self.approximate_pairwise_occurrence(photo_dict)
            
        out_dict = {'image_count': self.image_count,
                    'comatrix_images': self.comatrix,
                    'vocab_frequency_vector': self.vocab_frequency_vector}
        
        self.out_q.put(out_dict)


class PatternCountProcess(GetDatasetProcess):
    def __init__(self, filepath_q, out_q, init_dict, max_memory_percent, total_num_processes, out_dir, save_description):
        GetDatasetProcess.__init__(self, filepath_q, init_dict, False, True,
                                   max_memory_percent, total_num_processes, out_dir, save_description)

        self.vocab_frequency = collections.defaultdict(set) #set of occurrences of each concept

        #Final output
        self.out_q = out_q

    def freq_pattern_count(self, photo_dict):
        batch_patterns, batch_keys = self.concept_vocab_feature(photo_dict)
        for photokey in photo_dict:
            if photokey in batch_keys:
                pattern = batch_patterns[batch_keys[photokey], :]
                owner = photo_dict[photokey]["owner"]
                pattern_indices = sparse.find(pattern)[1]
                for p in pattern_indices:
                    self.vocab_frequency[p].add(owner)

    def run(self):
        print "Starting " + self.name
        while True:    
            photo_dict, break_condition = self.get_next_photo_dict()
            if break_condition:
                break
            self.freq_pattern_count(photo_dict)
            
        out_dict = {'vocab_frequency': self.vocab_frequency}       
        self.out_q.put(out_dict)


class TransactionListProcess(GetDatasetProcess):
    def __init__(self, filepath_q, init_dict, max_memory_percent, total_num_processes, out_dir, save_description, lock):
        GetDatasetProcess.__init__(self, filepath_q, init_dict, False, True,
                                   max_memory_percent, total_num_processes, out_dir, save_description)

        self.transaction_list = []
        self.lock = lock

    def process_dict(self, photo_dict):
        for photo in photo_dict.values():
            concept_ind = ["{}".format(self.vocab_index[c]) for c in photo['concepts'] if c in self.concept_vocabulary]
            if len(concept_ind) > 0:
                self.transaction_list.append(",".join(concept_ind))

    def save_to_file(self, output_dictionary):
        self.lock.acquire()
        with open(os.path.join(self.out_dir, self.save_description + '.txt'), 'a') as fp:
            fp.write("\n")
            fp.write("\n".join(output_dictionary))
        self.lock.release()

    def get_out_dict(self):
        return self.transaction_list

    def clear_variables(self):
        self.transaction_list = []


def write_to_file(output_dictionary, data_dir, save_description, postfix, limit):
        if not limit is None:
            freq_concepts = [(key, len(output_dictionary[key])) for key in output_dictionary if
                             len(output_dictionary[key]) > limit]
        else:
            freq_concepts = [(key, len(output_dictionary[key])) for key in output_dictionary]
        freq_concepts = sorted(freq_concepts, key=lambda x: x[1], reverse=True)

        with open(os.path.join(data_dir, "concepts", save_description + postfix), 'w') as fp:
            for concept in freq_concepts:
                fp.write("{}\t{}\n".format(concept[0], concept[1]))


def merge_vocabulary_files(data_dir, temp_dir, min_num_images, save_description):
        print "Starting Merge Vocabulary Files"
        confirmed_vocab = set([])
        path = os.path.join(data_dir, "concepts", "{}_confirmed_vocab.txt".format(save_description))

        if not os.path.exists(path):
            candidate_vocab = collections.defaultdict(set)

            #Initially make list of vocab with more than min_num_image without storing counts
            (_, _, filenames) = os.walk(temp_dir).next()
            for filename in filenames:
                filepath = os.path.join(temp_dir, filename)
                with open(filepath, 'r') as f:
                    temp_dict = json.load(f)

                for key in temp_dict['concept_owners']:
                    if not key in confirmed_vocab:
                        if len(temp_dict['concept_owners'][key]) > min_num_images:
                            confirmed_vocab.add(key)
                        else:
                            candidate_vocab[key].update(temp_dict['concept_owners'][key])
                            if len(candidate_vocab[key]) > min_num_images:
                                confirmed_vocab.add(key)
                                del candidate_vocab[key]

            candidate_vocab.clear()
            with open(path, 'w') as fin:
                fin.write("\n".join(confirmed_vocab))
        else:
            with open(path, 'r') as fin:
                confirmed_vocab = fin.read().split("\n")

        #Count statistics for confirmed_vocab
        photoID = set([])
        ownerID = set([])
        concepts_per_image = []
        concept_owners = collections.defaultdict(set)
        concept_images = collections.defaultdict(set)
        query_owners = collections.defaultdict(set)
        query_images = collections.defaultdict(set)

        (_, _, filenames) = os.walk(temp_dir).next()
        for filename in filenames:
            filepath = os.path.join(temp_dir, filename)
            with open(filepath, 'r') as f:
                temp_dict = json.load(f)

            photoID.update(temp_dict['photoID'])
            ownerID.update(temp_dict['ownerID'])
            concepts_per_image.extend(temp_dict['concepts_per_image'])
            for key in temp_dict['concept_owners']:
                if key in confirmed_vocab:
                    concept_owners[key].update(temp_dict['concept_owners'][key])
                    concept_images[key].update(temp_dict['concept_images'][key])

            for key in temp_dict['query_owners']:
                query_owners[key].update(temp_dict['query_owners'][key])
                query_images[key].update(temp_dict['query_images'][key])

        #Do before exit
        write_to_file(concept_owners, data_dir, save_description, "_owner_per_concept.txt", min_num_images)
        write_to_file(concept_images, data_dir, save_description, "_image_per_concept.txt", min_num_images)
        write_to_file(query_owners, data_dir, save_description, "_owner_per_query.txt", None)
        write_to_file(query_images, data_dir, save_description, "_image_per_query.txt", None)

        #Report Stats
        path = os.path.join(data_dir, "concepts", "{}_stats.txt".format(save_description))
        with open(path, 'w') as fin:
            fin.write('Number of images: {}\n'.format(len(photoID)))
            fin.write('Number of owners: {}\n'.format(len(ownerID)))
            fin.write('Mean number of concepts: {}\n'.format(np.mean(concepts_per_image)))
            fin.write('Median number of concepts: {}\n'.format(np.median(concepts_per_image)))
            fin.write('Q1 number of concepts: {}\n'.format(np.percentile(concepts_per_image, 25)))
            fin.write('Q3 number of concepts: {}\n'.format(np.percentile(concepts_per_image, 75)))
            fin.write('Std number of concepts: {}\n'.format(np.std(concepts_per_image)))
            fin.write('Max number of concepts: {}\n'.format(max(concepts_per_image)))
            fin.write('Min number of concepts: {}'.format(min(concepts_per_image)))

        print "Done with vocab file merge"


def merge_pairwise_occurrence_files(temp_dir, stats_dir, vis_dir, num_vocab, num_query, concept_vocabulary, query_vocabulary, save_description):
        print "Starting Merge Pairwise Files"
        out_dict_aggregate = {'codict_owner': collections.defaultdict(set),
            'codict_images': collections.defaultdict(set),
            'codict_query': collections.defaultdict(set),
            'vocab_frequency': set([]),
            'vocab_images': set([])}
        owner_set_aggregate = set([])
        query_frequency_aggregate = collections.defaultdict(set)

        pair_dir = os.path.join(vis_dir, "pair")
        if not os.path.exists(pair_dir):
            os.makedirs(pair_dir)
        single_dir = os.path.join(vis_dir, "single")
        if not os.path.exists(single_dir):
            os.makedirs(single_dir)

        #Do owner_set first because easiest
        save_dir = os.path.join(temp_dir, "owners_set")
        (_, _, filenames) = os.walk(save_dir).next()
        for filename in filenames:
            with open(os.path.join(save_dir, filename), 'r') as fp:
                owner_set_aggregate.update(fp.read().split('\n'))

        num_owners = len(owner_set_aggregate)
        del owner_set_aggregate

        #Do query_frequency
        save_dir = os.path.join(temp_dir, "query_frequency")
        (_, _, filenames) = os.walk(save_dir).next()
        for filename in filenames:
            with open(os.path.join(save_dir, filename), 'r') as fp:
                json_dict = json.load(fp)
            for key in json_dict:
                query_frequency_aggregate[key].update(json_dict[key])

        keys = query_frequency_aggregate.keys()
        vals = [len(v) for v in query_frequency_aggregate.values()]
        query_freq_vect = np.zeros((num_query, 1))
        query_freq_vect[keys, 0] = vals

        del query_frequency_aggregate

        #The rest are aggregated on a per vocab basis
        save_dir = os.path.join(temp_dir, "by_vocab")
        comatrix_owners = sparse.lil_matrix((num_vocab, num_vocab))
        comatrix_images = sparse.lil_matrix((num_vocab, num_vocab))
        comatrix_query = sparse.lil_matrix((num_vocab, num_vocab))
        vocab_freq_vect = np.zeros((num_vocab, 1))
        for (dir, _, filenames) in os.walk(save_dir):
            if len(filenames) > 0:
                vocab = int(dir.split("\\")[-1])
                for filename in filenames:
                    filepath = os.path.join(dir, filename)
                    with open(filepath, 'r') as f:
                        json_dict = json.load(f)

                    out_dict_aggregate["vocab_frequency"].update(json_dict["vocab_frequency"])
                    out_dict_aggregate["vocab_images"].update(json_dict["vocab_images"])

                    for key in json_dict["codict_owner"]:
                        out_dict_aggregate["codict_owner"][key].update(json_dict["codict_owner"][key])
                        out_dict_aggregate["codict_images"][key].update(json_dict["codict_images"][key])
                        out_dict_aggregate["codict_query"][key].update(json_dict["codict_query"][key])

                #Add entries to co-occurrence matrices
                keys = out_dict_aggregate["codict_owner"].keys()
                if len(keys) > 0:
                    vals = [len(v) for v in out_dict_aggregate["codict_owner"].values()]
                    vocab_vector = vocab * np.ones((len(keys), 1))
                    comatrix_owners[vocab_vector, keys] = vals

                    keys = [int(float(x)) for x in out_dict_aggregate["codict_images"].keys()]
                    vals = [len(v) for v in out_dict_aggregate["codict_images"].values()]
                    comatrix_images[vocab_vector, keys] = vals

                    keys = [int(float(x)) for x in out_dict_aggregate["codict_query"].keys()]
                    vals = [len(v) for v in out_dict_aggregate["codict_query"].values()]
                    comatrix_query[vocab_vector, keys] = vals

                vocab_freq_vect[vocab, 0] = len(out_dict_aggregate["vocab_frequency"])

                #Save images for visualization
                save_dir = os.path.join(single_dir, "{}.txt".format(vocab))
                with open(save_dir, 'w') as fp:
                    fp.write("\n".join(list(out_dict_aggregate["vocab_images"])))

                save_dir = os.path.join(pair_dir, "{}.txt".format(vocab))
                with open(save_dir, 'w') as fp:
                    json.dump({k: list(out_dict_aggregate["codict_images"][k]) for k in out_dict_aggregate["codict_images"]}, fp)

                #Clear variables
                out_dict_aggregate["codict_owner"].clear()
                out_dict_aggregate["codict_images"].clear()
                out_dict_aggregate["codict_query"].clear()
                out_dict_aggregate["vocab_frequency"].clear()
                out_dict_aggregate["vocab_images"].clear()


        path = os.path.join(stats_dir, "{}_flickr_comatrix_low_mem.mat".format(save_description))
        spio.savemat(path, {'comatrix_owners': comatrix_owners,
                            'comatrix_images': comatrix_images,
                            'comatrix_queries': comatrix_query,
                            'concept_vocabulary': concept_vocabulary,
                            'query_vocabulary': query_vocabulary,
                            'vocab_owner_count': vocab_freq_vect,
                            'query_owner_count': query_freq_vect,
                            'total_owners': num_owners})

        print "Done generating co-occurrence matrices"
        return


class MergeApproximatePairwiseOccurrenceThread(Thread):
    def __init__(self, name, photo_queue, stats_dir, save_description, concept_vocabulary):
        Thread.__init__(self)
        #Inputs
        self.name = name
        self.photo_queue = photo_queue
        self.stats_dir = stats_dir
        self.save_description = save_description
        self.concept_vocabulary = concept_vocabulary
        self.num_vocab = len(concept_vocabulary)

    def safe_exit(self):
        #Add a poison pill
        self.photo_queue.put("Stop")

    def run(self):
        print "Starting " + self.name
        image_count = 0
        comatrix_images = sparse.csr_matrix((self.num_vocab, self.num_vocab))
        vocab_freq_vect = np.zeros((self.num_vocab, 1))

        while True:
            photo_dict = self.photo_queue.get()

            if photo_dict == "Stop":
                # Poison pill shutdown
                print "Exiting " + self.name
                break

            #print photo_dict['image_count']
            image_count += photo_dict['image_count']
            comatrix_images = comatrix_images + photo_dict['comatrix_images']
            vocab_freq_vect = vocab_freq_vect + photo_dict['vocab_frequency_vector']

        ##############################################################
        #Do before exit
        ##############################################################

        #Save co-occurrence matrix
        path = os.path.join(self.stats_dir, "{}_flickr_comatrix_approx.mat".format(self.save_description))
        spio.savemat(path, {'comatrix_images': comatrix_images,
                            'vocab_image_count': vocab_freq_vect,
                            'total_images': image_count,
                            'concept_vocabulary': self.concept_vocabulary})

        print "Done generating co-occurrence matrices"
        return


class MergePatternCountThread(Thread):
    def __init__(self, name, photo_queue, data_dir, knowledge_dir, save_description, min_num_images, concept_vocabulary):
        Thread.__init__(self)

        #Input variables
        self.name = name
        self.data_dir = data_dir
        self.knowledge_dir = knowledge_dir
        self.save_description = save_description
        self.photo_queue = photo_queue
        self.min_num_images = min_num_images
        self.concept_vocabulary = concept_vocabulary
        self.output_vocabulary = []

    def safe_exit(self):
        #Add a poison pill
        self.photo_queue.put("Stop")

    def write_to_file(self, output_dictionary, postfix, limit):
        if not limit is None:
            freq_concepts = [(key, len(output_dictionary[key])) for key in output_dictionary if
                             len(output_dictionary[key]) > limit]
        else:
            freq_concepts = [(key, len(output_dictionary[key])) for key in output_dictionary]
        freq_concepts = sorted(freq_concepts, key=lambda x: x[1], reverse=True)

        with open(os.path.join(self.data_dir, "concepts", self.save_description + postfix), 'w') as fp:
            for concept in freq_concepts:
                fp.write("{}\t{}\n".format(concept[0], concept[1]))

    def run(self):
        print "Starting " + self.name

        vocab_frequency_dict = collections.defaultdict(set)  #vector of occurrences of each concept
        while True:
            photo_dict = self.photo_queue.get()

            if photo_dict == "Stop":
                # Poison pill shutdown
                print "Exiting " + self.name
                break

            for key in photo_dict['vocab_frequency'].keys():
                vocab_frequency_dict[key].update(photo_dict['vocab_frequency'][key])

        #Do before exit
        vocab_vector = np.zeros((len(self.concept_vocabulary), ))
        vocab_vector[vocab_frequency_dict.keys()] = [len(vocab_frequency_dict[key]) for key in vocab_frequency_dict.keys()]

        supported_vocab_ind = np.where(vocab_vector > self.min_num_images)[0]
        excluded_vocab_ind  = np.where(vocab_vector <= self.min_num_images)[0]

        supported_vocab = sorted([(self.concept_vocabulary[ii], vocab_vector[ii]) for ii in supported_vocab_ind], key=lambda x: x[1], reverse=True)
        excluded_vocab = sorted([(self.concept_vocabulary[ii], vocab_vector[ii]) for ii in excluded_vocab_ind], key=lambda x: x[1], reverse=True)

        supported_vocab_formated = ["{}\t{}".format(t[0], t[1]) for t in supported_vocab]
        save_path = os.path.join(self.data_dir, 'concepts', self.save_description + "_newest.txt")
        with open(save_path, 'a') as f:
            f.write("\n".join(supported_vocab_formated))
            f.write("\n")

        excluded_vocab_formated = ["{}\t{}".format(t[0], t[1]) for t in excluded_vocab]
        save_path = os.path.join(self.knowledge_dir, "Freebase", "candidate_concepts", self.save_description + "_no_support.txt")
        with open(save_path, 'a') as f:
            f.write("\n".join(excluded_vocab_formated))
            f.write("\n")

        #Return value
        self.output_vocabulary =  supported_vocab


def find_vocabulary(data_dir, stats_dir, category, min_num_images, save_description):
    print "Start find vocabulary"
    filequeue = JoinableQueue()
    photoqueue = Queue()

    init_dict = initialize_variables(None, None, False)

    # Create new processes
    num_processes = cpu_count()
    temp_dir = os.path.join(stats_dir, "database_temp", "vocab", category)
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    processes = [FindVocabularyProcess(filequeue, photoqueue, init_dict, 30.0, num_processes, temp_dir, category) for i in xrange(num_processes)]
    for p in processes:
        p.start()

    #Add the files to the process queue
    add_files_to_queue(data_dir, category, filequeue)
    #Add a poison pill for each process
    for i in xrange(num_processes):
        filequeue.put("Stop")

    for p in processes:
        p.join()

    merge_vocabulary_files(data_dir, temp_dir, min_num_images, save_description)

    print "Removing temp files"
    shutil.rmtree(temp_dir)

    print "Done with find vocabulary"


def find_concept_pairs(data_dir, stats_dir, vis_dir, category, concept_vocabulary):
    print "Starting Pairwise Calculation"
    filequeue = JoinableQueue()

    category_file = os.path.join(data_dir, 'categories', category + '.txt.')
    with open(category_file, 'r') as fp:
        query_vocabulary = json.load(fp).keys()

    concept_vocabulary_list, concept_vocabulary_freq = zip(*concept_vocabulary)
    init_dict = initialize_variables(concept_vocabulary_list, query_vocabulary, True)

    # Create new processes
    num_processes = cpu_count()
    lock = Lock()
    temp_dir = os.path.join(stats_dir, "database_temp", "pairwise", category)
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    processes = [PairwiseOccurrenceProcess(filequeue, init_dict, temp_dir, category, num_processes, 30.0, lock)
                 for i in xrange(num_processes)]

    for p in processes:
        p.start()

    #Add the files to the process queue
    add_files_to_queue(data_dir, category, filequeue)
    #Add a poison pill for each process
    for i in xrange(num_processes):
        filequeue.put("Stop")

    for p in processes:
        p.join()

    merge_pairwise_occurrence_files(temp_dir, stats_dir, vis_dir, len(concept_vocabulary), len(query_vocabulary),
                                    concept_vocabulary_list, query_vocabulary, category)
    print "Removing temp files"
    shutil.rmtree(temp_dir)

    print "Done with Pairwise Calculation"


def find_approximate_concept_pairs(data_dir, stats_dir, category, save_description, concept_vocabulary):
    print "Start approximate count"
    filequeue = JoinableQueue()
    photoqueue = Queue()

    category_file = os.path.join(data_dir, 'categories', category + '.txt.')
    with open(category_file, 'r') as fp:
        query_vocabulary = json.load(fp).keys()

    concept_vocabulary_list, concept_vocabulary_freq = zip(*concept_vocabulary)
    init_dict = initialize_variables(concept_vocabulary_list, query_vocabulary, True)

    # Create merge thread
    # If processes finish at different times, the thread can already start merging the variables
    mergethread = MergeApproximatePairwiseOccurrenceThread('Merge-Thread', photoqueue, stats_dir, save_description, concept_vocabulary_list)
    mergethread.start()

    # Create new processes
    num_processes = cpu_count()
    processes = [ApproxPairwiseOccurrenceProcess(filequeue, photoqueue, init_dict, 30.0, num_processes, "", category) for i in xrange(num_processes)]
    for p in processes:
        p.start()

    #Add the files to the process queue
    add_files_to_queue(data_dir, category, filequeue)
    #Add a poison pill for each process
    for i in xrange(num_processes):
        filequeue.put("Stop")

    for p in processes:
        p.join()

    mergethread.safe_exit()
    mergethread.join()

    print "Done with Approximate Pairwise Calculation"


def freq_pattern_count(knowledge_dir, data_dir, stats_dir, category, save_description, concept_vocabulary, min_num_images):
    print "Start frequent pattern count"
    filequeue = JoinableQueue()
    photoqueue = Queue()

    concept_vocabulary_list, concept_vocabulary_freq = zip(*concept_vocabulary)
    init_dict = initialize_variables(concept_vocabulary_list, None, True)

    # Create merge thread
    # If processes finish at different times, the thread can already start merging the variables
    mergethread = MergePatternCountThread('Merge-Thread', photoqueue, data_dir, knowledge_dir, save_description, min_num_images, concept_vocabulary_list)
    mergethread.start()

    # Create new processes
    num_processes = cpu_count()
    temp_dir = os.path.join(stats_dir, "database_temp", "vocab_extended", category)
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    processes = [PatternCountProcess(filequeue, photoqueue, init_dict, 30, num_processes, temp_dir, category) for i in xrange(num_processes)]
    for p in processes:
        p.start()

    #Add the files to the process queue
    add_files_to_queue(data_dir, category, filequeue)
    #Add a poison pill for each process
    for i in xrange(num_processes):
        filequeue.put("Stop")

    for p in processes:
        p.join()

    mergethread.safe_exit()
    mergethread.join()

    print "Done with frequent pattern count"
    return mergethread.output_vocabulary


def save_transaction_list(data_dir, stats_dir, category, concept_vocabulary, save_description):
    print "Start saving transaction list"
    filequeue = JoinableQueue()

    concept_vocabulary_list, concept_vocabulary_freq = zip(*concept_vocabulary)
    init_dict = initialize_variables(concept_vocabulary_list, None, True)

    # Create new processes
    temp_dir = os.path.join(stats_dir, "transaction_list")
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    else:
        print "todo"
    lock = Lock()
    num_processes = cpu_count()
    processes = [TransactionListProcess(filequeue, init_dict, 30, num_processes, temp_dir, save_description, lock) for i in xrange(num_processes)]
    for p in processes:
        p.start()

    #Add the files to the process queue
    add_files_to_queue(data_dir, category, filequeue)
    #Add a poison pill for each process
    for i in xrange(num_processes):
        filequeue.put("Stop")

    for p in processes:
        p.join()

    print "Removing temp files"
    shutil.rmtree(temp_dir)

    print "Done with saving transaction list"


def initialize_variables(concept_vocabulary_list, query_vocabulary, has_vocabulary):
    init_dict = {}

    # Method specific variables
    if has_vocabulary:
        concept_vocabulary_no_spaces = ["".join(w.split()) for w in concept_vocabulary_list]

        #Inputs
        init_dict['concept_vocabulary'] = concept_vocabulary_list
        init_dict['num_vocab'] = len(concept_vocabulary_list)
        init_dict['vocab_index'] = dict(zip(init_dict['concept_vocabulary'], range(0, init_dict['num_vocab'])))

        init_dict['query_vocabulary'] = query_vocabulary
        if not query_vocabulary is None:
            init_dict['num_query'] = len(query_vocabulary)
            init_dict['query_index'] = dict(zip(init_dict['query_vocabulary'], range(0, init_dict['num_query'])))
        else:
            init_dict['num_query'] = 0
            init_dict['query_index'] = {}

        init_dict['wordlist'] = list(set(concept_vocabulary_no_spaces +[ w for t in concept_vocabulary_list for w in t.split()]))
        init_dict['num_words'] = len(init_dict['wordlist'])
        init_dict['wordlist_index'] = dict(zip(init_dict['wordlist'], range(0, init_dict['num_words'])))

        #Features
        init_dict['num_words_in_vocab'] = np.zeros((init_dict['num_vocab'], 1))
        init_dict['vocab_feature'] = sparse.lil_matrix((init_dict['num_vocab'], init_dict['num_words']))
        init_dict['synonym_feature'] = sparse.lil_matrix((init_dict['num_vocab'], init_dict['num_vocab']))
        for vocabInd in range(0, init_dict['num_vocab']):
            pattern = concept_vocabulary_list[vocabInd]
            pattern_delim = pattern.split()
            init_dict['num_words_in_vocab'][vocabInd] = len(pattern_delim)

            #A phrase is always the same as itself, with or without spaces
            wordInd = init_dict['wordlist_index'][pattern.replace(' ', '')]
            init_dict['vocab_feature'][vocabInd, wordInd] = 1
            init_dict['synonym_feature'][vocabInd, vocabInd] = 1

            #A phrase is always the same as the words composing the phrase
            wordInd = [init_dict['wordlist_index'][p] for p in pattern_delim]
            init_dict['vocab_feature'][vocabInd, wordInd] = 1.0/len(pattern_delim)
            #However the component words should only be counted as part of the phrase, not individual words
            if len(pattern_delim) > 1:
                wordInd = [init_dict['vocab_index'][p] for p in pattern_delim if p in init_dict['concept_vocabulary']]
                init_dict['synonym_feature'][wordInd, vocabInd] = -1

        init_dict['vocab_feature'] = sparse.csr_matrix(init_dict['vocab_feature'])
        init_dict['synonym_feature'] = sparse.csr_matrix(init_dict['synonym_feature'])

    return init_dict


def add_files_to_queue(data_dir, category, filequeue):

    #Get category queries
    category_file = os.path.join(data_dir, 'categories', category + '.txt.')
    with open(category_file, 'r') as fp:
        query_list = json.load(fp).keys()

    #Fill queue
    for query in query_list:
        starting_dir = os.path.join(data_dir, 'texts', query)
        if os.path.exists(starting_dir):
            for dirName, subdirList, fileList in os.walk(starting_dir):
                #print('Number of files to go: %d' % filequeue.qsize())
                #print('Found directory: %s' % dirName)

                for fname in fileList:
                    filepath = os.path.join(starting_dir, dirName, fname)
                    filequeue.put(filepath)


if __name__ == '__main__':
    freeze_support()
    with open("E:\\test_freq_pattern_count.json", 'r') as f:
        json_dict = json.load(f)

    json_dict['filtered_candidates'] = [t[0] for t in json_dict['filtered_candidates']]
    freq_pattern_count(json_dict['knowledge_dir'], json_dict['root'], json_dict['query'], json_dict['filtered_candidates'], json_dict['min_num_images'])