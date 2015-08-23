import os, sys, time, signal, threading
import json, urllib.request, urllib.parse, urllib.error
from math import floor
import crawler.flickr_threads_toSQL as mtcwl
from nltk.corpus import stopwords
import crawler.config as cfg
from crawler.flickrapi2 import FlickrAPI
from multiprocessing import Process, freeze_support, Queue
from queue import Empty

def get_num_photos(query, flickrAPIKey):
    num_images = -1
    query = query.encode('ascii', errors='ignore')
    try:
        rsp = fapi.photos_search(api_key=flickrAPIKey, ispublic="1", media="photos", text=query)
        fapi.testFailure(rsp)
    except:
        print("Error in get_num_photos: counts will be inaccurate")
    else:
        if getattr(rsp, 'photos', None):
            if getattr(rsp.photos[0], 'attrib', None):
                num_images = int(rsp.photos[0].attrib["total"])
    return num_images

def start_crawler(config, category, num_images, communication_q, rate_limit):
    c = mtcwl.MultiThreadedFlickrCrawler(config, category, num_images, communication_q, rate_limit)
    c.start()

# Paging example from Freebase API
def do_query(query_url, cursor=""):
    if cursor == "":
        url = query_url + '&cursor'
    else:
        url = query_url + '&cursor=' + cursor
    response = json.loads(urllib.request.urlopen(url).read())
    return response["cursor"], response

def signal_handler(proc, communication_q):
    input_str = input("Type 'quit' to stop download")
    print(input_str)
    while input_str != "quit":
        input_str = input("Type 'quit' to stop download")

    for p in proc:
        communication_q.put("exit")
        print((p.name))

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Use: run_download_multiple_query.py freebase_type min_images')
        sys.exit()

    freebase_type = sys.argv[1]  # "/biology/domesticated_animal"
    if freebase_type[0] == "/":
        freebase_type = freebase_type[1:]
    min_images = int(sys.argv[2])

    save_category = 'E:\\rooms\categories\\' + freebase_type.replace('/', '_') + '.txt'
    if(os.path.exists(save_category)):
        with open(save_category, 'r') as f:
            freq_categories = json.load(f)
    else:
        freebase_api_key = cfg.cfg.vars["freebaseAPIKey"]
        service_url = 'https://www.googleapis.com/freebase/v1/mqlread'
        query = [{"id": None, "name": None, "type": freebase_type, "limit":100}]

        params = {'key': freebase_api_key, 'query': json.dumps(query)}
        url = service_url + '?' + urllib.parse.urlencode(params)

        cursor, response = do_query(url)
        while cursor:
            cursor, next_response = do_query(url, cursor)
            response["result"] += next_response["result"]

        flickrAPIKey = cfg.cfg.vars["flickrAPIkeys"].split(', ')[0]
        flickrSecret = cfg.cfg.vars["flickrAPIsecrets"].split(', ')[0]
        fapi = FlickrAPI(flickrAPIKey, flickrSecret)
        freq_categories = {}
        for category in response["result"]:
            if ("name" in category) and not(category["name"] is None):
                category_name = category["name"].lower()
                num_images = get_num_photos(category_name, flickrAPIKey)
                if num_images > min_images:
                    freq_categories[category_name] = num_images
            # else:
            #     # Try with the noun from last word in name
            #     category_name_delim = category_name.split()
            #     category_name_noun = category_name_delim[-1]
            #     if not category_name_noun in freq_categories:
            #         num_images = get_num_photos(category_name_noun, flickrAPIKey)
            #         if num_images > min_images:
            #             freq_categories[category_name_noun] = num_images
            with open(save_category, 'w') as f:
                json.dump(freq_categories, f)

    remove_keys = ['cat', 'tre', 'willow', 'pony', 'camel', 'fish', 'manufacturing', 'construction', 'industry', 'transport', 'equipment', 'slide', 'organic']
    for key in remove_keys:
        if key in freq_categories:
            del freq_categories[key]

    proc = []
    communication_q = Queue()
    crawlers = []

    rate_limit = floor(3600 * (len(cfg.cfg.vars["flickrAPIkeys"].split())-1) / len(freq_categories))
    for category in freq_categories:
        p = Process(target=start_crawler, args=(cfg.cfg, category, 1500000, communication_q, rate_limit))
        proc.append(p)

    #Exit safely
    # exit_thread = threading.Thread(target=signal_handler, args=(proc, communication_q))
    # exit_thread.daemon = True
    # exit_thread.start()

    for p in proc:
        p.start()

    for p in proc:
        p.join()

    print("Main thread exiting")
    sys.exit()


