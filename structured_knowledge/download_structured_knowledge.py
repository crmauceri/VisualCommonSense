__author__ = 'mauceri2'
import os
import json
import urllib
import time
from multiprocessing import Process, Lock, Queue
from math import ceil
from core.stl import getFileOfType
from database_builder.core.timing import tic
from database_builder.core.timing import toc

class curl_Freebase_Thread(Process):
    def __init__(self, error_list, lock, api_key, concepts, startind, stopind, save_dir, redownload):
        Process.__init__(self)
        self.error_list = error_list
        self.lock = lock
        self.api_key = api_key
        self.concepts = concepts
        self.startind = startind
        self.stopind = stopind
        self.save_dir = save_dir
        self.redownload = redownload

    def run(self):
        #print "Starting Freebase {} through {}\n".format(self.startind, self.stopind)
        search_url = 'https://www.googleapis.com/freebase/v1/search'
        topic_url = "https://www.googleapis.com/freebase/v1/topic"
        topic_params = {'key': self.api_key}
        for concept in self.concepts[self.startind:self.stopind]:
            if concept.replace(" ", "").isalpha():
                composite = []
                save_path = os.path.join(self.save_dir, concept + '.json')
                if (not os.path.exists(save_path)) or self.redownload:
                    search_params = {'key': self.api_key, 'query': concept}
                    url = search_url + '?' + urllib.urlencode(search_params)
                    search_json = json.loads(urllib.urlopen(url).read())
                    if("result" in search_json):
                        maxResult = min(5, len(search_json["result"]))
                        for result in search_json["result"][0:maxResult]:
                            if "id" in result:
                                topic_id = result["id"]
                                url = topic_url + topic_id + '?' + urllib.urlencode(topic_params)
                                try:
                                    concept_json = urllib.urlopen(url).read()
                                    composite.append((topic_id, json.loads(concept_json)))
                                except IOError as E:
                                    print "Error on {}".format(save_path)
                                    self.lock.acquire()
                                    self.error_list.append(concept)
                                    self.lock.release()

                    try:
                        with open(save_path, 'w') as fin:
                            json.dump(composite, fin)
                    except:
                        print "{} Freebase error on {}".format(self.name, save_path)
                        self.lock.acquire()
                        self.error_list.append(concept)
                        self.lock.release()
        #print "Done with Freebase {} through {}\n".format(self.startind, self.stopind)

class curl_ConceptNet_Thread(Process):
    limit = 10000 #If there are many server errors, try decreasing the limit.

    def __init__(self, error_list, lock, concepts, startind, stopind, save_dir, redownload, rate_limit):
        Process.__init__(self)
        self.error_list = error_list
        self.rate_limit = rate_limit
        self.lock = lock
        self.concepts = concepts
        self.startind = startind
        self.stopind = stopind
        self.save_dir = save_dir
        self.redownload = redownload

    def run(self):
        #print "Starting ConceptNet {} through {}\n".format(self.startind, self.stopind)
        top_url = "http://conceptnet5.media.mit.edu/data/5.2/search?text="
        for concept in self.concepts[self.startind:self.stopind]:
            if concept.replace(" ", "").isalpha():
                concept = concept.replace(" ", "_")
                save_path = os.path.join(self.save_dir, concept + '.json')
                if (not os.path.exists(save_path)) or self.redownload:
                    url = "{}{}&limit={}".format(top_url, concept, self.limit)

                    try:
                        concept_json = urllib.urlopen(url).read()
                        timer = time.time() #Start timer
                        if not 'error' in json.loads(concept_json):
                            with open(save_path, 'w') as fin:
                                fin.write(concept_json)
                        else:
                            print('{} Query error on {}').format(self.name, concept)
                            self.lock.acquire()
                            self.error_list.append(concept)
                            self.lock.release()
                            start = time.time()
                            time.sleep(60)
                            print time.time() - start
                    except IOError as e:
                        print "{} IOError on {}:{}".format(self.name, save_path, e)
                        self.lock.acquire()
                        self.error_list.append(concept)
                        self.lock.release()
                    except ValueError as e:
                        print "{} ValueError on {}:{}".format(self.name, save_path, e)
                        self.lock.acquire()
                        self.error_list.append(concept)
                        self.lock.release()

                    #Sleep for remainder of rate limit
                    time_elapsed = time.time() - timer
                    time.sleep(self.rate_limit-time_elapsed)

        #print "Done with ConceptNet {} through {}\n".format(self.startind, self.stopind)

def download_structured_knowledge(knowledge_root, concept_list, do_skip):
    #Make save directories if nessisary
    freebase_dir = os.path.join(knowledge_root, 'FreebaseJSON')
    if not os.path.exists(freebase_dir):
        os.makedirs(freebase_dir)

    concept_net_dir = os.path.join(knowledge_root, 'ConceptNetJSON')
    if not os.path.exists(concept_net_dir):
        os.makedirs(concept_net_dir)

    #Let's make some threads
    freebase_api_key = "AIzaSyD_KwyBN0SW5DnhBzFlyECbIG7fWbsuWI8"
    error_list = []
    lock = Lock()
    num_concepts = len(concept_list)

    num_processes = 16 #Estimated number of processes
    concept_inc = range(0, num_concepts, max(num_concepts/num_processes, 1))
    if(not concept_inc[-1] == num_concepts-1):
        concept_inc.append(num_concepts-1)

    num_processes = len(concept_inc)-1 #Actual number of processes
    concept_net_rate_limit = (60.0*num_processes)/ 900.0
    threads = []
    for i in range(0, num_processes):
        startind = concept_inc[i]
        stopind = concept_inc[i+1]

        thread = curl_Freebase_Thread(error_list, lock, freebase_api_key, concept_list, startind, stopind, freebase_dir, not do_skip)
        thread.start()
        threads.append(thread)

        thread = curl_ConceptNet_Thread(error_list, lock, concept_list, startind, stopind, concept_net_dir, not do_skip, concept_net_rate_limit)
        thread.start()
        threads.append(thread)

    for t in threads:
        t.join()

    trys = 0
    while len(error_list) > 0 and trys < 5:
        print "Redownloading errors"
        error_list = download_structured_knowledge(knowledge_root, error_list, True)
        trys += 1

    return error_list

if __name__ == "__main__":
    import sys

    save_description = sys.argv[3]
    concept_dir = sys.argv[2]
    concept_file = os.path.join(concept_dir, '{}_owner_per_concept.txt'.format(save_description))
    with open(concept_file, 'r') as f:
        concept_vocabulary = [t.split()[0] for t in f.read().split('\n') if len(t) > 0]

    download_structured_knowledge(sys.argv[1], concept_vocabulary, True)

