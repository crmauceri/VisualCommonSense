#!/usr/bin/python

# Image querying script written by Tamara Berg,
# and extended heavily James Hays
# Modified by Juan C. Caicedo on Jan. 2013
# Further modified by Cecilia Mauceri Feb 2015

import sys, string, math, time, socket
import random, os, re
import threading
from multiprocessing import Queue, JoinableQueue
from queue import Empty
import subprocess
import crawler.config
from crawler.flickrapi2 import FlickrAPI, FlickrExpatError
from datetime import datetime
from crawler.dbdirs import DBDirectories
from crawler.download_thread_toSQL import DownloadImageThread, simpleDateFormat
import xml.parsers.expat
from requests.exceptions import SSLError

class MultiThreadedFlickrCrawler:
    ###########################################################################
    # System parameters and initializations
    ###########################################################################
    def __init__(self, cfg, category, max_num_images, communication_q, rate_limit):
        self.cfg = cfg
        self.category = category
        argv = self.cfg.vars
        self.communication_q = communication_q
        self.do_exit = False
        self.rate_limit = rate_limit
        self.rate_q = Queue()

        # flickr auth information: change these to your flickr api keys and secret
        self.flickrAPIkeys = argv["flickrAPIkeys"].split(', ')  # API key
        self.flickrAPIsecrets = argv["flickrAPIsecrets"].split(', ')  # shared "secret"
        self.queryFileName = argv["queryFileName"]  #'query_terms.txt'
        self.homeDir = argv["homeDir"]
        self.imagesPerDir = int(argv["imagesPerDir"])
        self.flickrerrors = 0

        # Crawler parameters
        self.resultsPerPage = int(argv["resultsPerPage"])
        self.downloadsPerQuery = int(argv["downloadsPerQuery"])
        self.numberOfThreads = int(argv["numberOfThreads"])
        self.startTime = int(argv["crawlerBeginTime"])  #1072915200 # 01/01/2004
        self.finalTime = int(time.time())
        self.singleDay = 86400  # 24hr*60min*60sec = 1day
        self.max_num_images = max_num_images
        self.database = argv["databaseName"]

        # Structures Initialization
        self.dbdir = DBDirectories(self.homeDir, argv["sysdir"], category)
        self.indexOfUniqueImages = self.dbdir.inf + 'imageIndex.txt'
        self.indexOfUniqueUsers = self.dbdir.inf + 'usersIndex.txt'
        self.recentUsers = dict()
        self.queryTerms = []

        # Multithreaded downloading of images
        self.queue = JoinableQueue()
        self.out_queue = JoinableQueue()
        self.threadsList = []
        for i in range(self.numberOfThreads):
            t = DownloadImageThread(self.queue, self.out_queue, self.dbdir.img, self.dbdir.txt, self.category,
                                    self.database)
            t.setDaemon(True)
            t.start()
            self.threadsList.append(t)

        print(("{} initialized".format(self.category)))

    ###########################################################################
    # Method to load query terms
    ###########################################################################
    def loadQueries(self):
        # Each term is a category
        self.queryTerms = [self.category]
        print(('positive queries:', self.queryTerms))
        list(map(lambda t: t.setValidTags(self.queryTerms), self.threadsList))
        return len(self.queryTerms)

    ###########################################################################
    # Method to load index of image names
    ###########################################################################
    def loadImageNamesIndex(self):
        print('Loading index of images')
        if os.path.exists(self.indexOfUniqueImages):
            self.allImageNames = dict(
                [(img.replace('\n', ''), True) for img in open(self.indexOfUniqueImages).readlines()])
            print(('Index with', len(self.allImageNames), 'names is ready to use'))
        else:
            self.allImageNames = dict()
            print(('No previous index found at {}'.format(self.indexOfUniqueImages)))
        print('Loading index of users')
        if os.path.exists(self.indexOfUniqueUsers):
            self.recentUsers = dict([(usr.replace('\n', ''), 1) for usr in open(self.indexOfUniqueUsers).readlines()])
            print(('Index with', len(self.recentUsers), 'users is ready to use'))
        else:
            self.recentUsers = dict()
            print(('No previous user index found at {}'.format(self.indexOfUniqueUsers)))

    ###########################################################################
    # Find out if an image is a duplicate or of a user already visited
    ###########################################################################
    def isDuplicateImage(self, flickrResult):
        b = flickrResult
        owner_date = b['owner'] + '_' + simpleDateFormat(b['datetaken'])
        imgName = b['server'] + '_' + b['id'] + '_' + b['secret'] + '_' + owner_date + '.jpg'
        alreadyIndexed = False
        userPhotos = 0

        if imgName in self.allImageNames:
            alreadyIndexed = self.allImageNames[imgName]
        else:
            self.allImageNames[imgName] = False

        if owner_date in self.recentUsers:
            userPhotos = self.recentUsers[owner_date]
        else:
            self.recentUsers[owner_date] = 0

        if (not alreadyIndexed) and userPhotos < 1:
            self.recentUsers[owner_date] += 1
            self.allImageNames[imgName] = True
            return False
        else:
            return True

    ###########################################################################
    #Find out if medium format of photo exists for download
    ###########################################################################
    def get_url(self, flickrResult, fapi, size):
        url = "https://farm{}.staticflickr.com/{}/{}_{}.jpg".format(flickrResult['farm'], flickrResult['server'], flickrResult['id'], flickrResult['secret'])
        return True, url

        #TODO find way to speed up actual url retrieval
        # image_id = flickrResult['id']
        # success = False
        # try:
        #     rsp = fapi.photos_getSizes(api_key=self.flickrAPIKey, photo_id=image_id)
        #     fapi.testFailure(rsp)
        # except:
        #      print sys.exc_info()[0]
        #      print ('Exception encountered while querying for urls\n')
        # else:
        #     if getattr(rsp, 'sizes', None):
        #         if int(rsp.sizes[0]['candownload']) == 1:
        #             if getattr(rsp.sizes[0], 'size', None):
        #                 for image_size in rsp.sizes[0].size:
        #                     if image_size['label'] == size:
        #                         return True, image_size['source']
        #
        # return False, ""

    ###########################################################################
    # Update index of unique image names
    ###########################################################################
    def updateImageNamesIndex(self, newImages):
        with open(self.indexOfUniqueImages, 'a') as indexFile:
            for img in newImages:
                indexFile.write(img + '\n')
        self.allImageNames = []

    ###########################################################################
    # Main Method. This runs the crawler in an infinite loop
    ###########################################################################
    def start(self):
        socket.setdefaulttimeout(30)  #30 second time out on sockets before they throw
        self.cfg.log(self.homeDir, "CRAWLER STARTED")
        while not self.do_exit:
            try:
                command = self.communication_q.get(False)
            except Empty as e:
                #Randomly choose flickrAPIkeys and flickrAPIsecrets
                currentKey = int(math.floor(random.random()*len(self.flickrAPIkeys)))
                # make a new FlickrAPI instance
                fapi = FlickrAPI(self.flickrAPIkeys[currentKey], self.flickrAPIsecrets[currentKey])
                num_queries = self.loadQueries()
                if num_queries == 0:
                    break
                newImages = []
                # Set time variables
                self.finalTime = int(time.time())
                currentTimeWindow = self.finalTime - self.startTime
                mintime = self.startTime + random.randint(0, currentTimeWindow)
                maxtime = mintime + 3 * self.singleDay
                print(('Since:', datetime.fromtimestamp(mintime)))
                print(('Until:', datetime.fromtimestamp(maxtime)))
                print(('Previous Users:', len(self.recentUsers)))
                self.loadImageNamesIndex()
                if len(self.allImageNames) > self.max_num_images:
                    print("Max Images reached")
                    break
                # Search Images using the query terms
                for current_tag in range(0, num_queries):
                    dirNumName = self.dbdir.uploadCurrentDirAndGetNext(self.imagesPerDir, self.queryTerms)
                    print(("Current Directory Number: ", dirNumName))
                    #form the query string.
                    query_string = self.queryTerms[current_tag]
                    print(('\n\nquery_string is ' + query_string))
                    #only visit 8 pages max, to try and avoid the dreaded duplicate bug.
                    #8 pages * 250 images  = 2000 images, should be duplicate safe.  Most interesting pictures will be taken.
                    num_visit_pages = 16
                    pagenum = 1
                    while ( pagenum <= num_visit_pages ):
                        if (self.rate_q.qsize()>self.rate_limit):
                            #Age out time stamps older than one hour
                            found_all = False
                            while(not found_all):
                                next_stamp = self.rate_q.get()
                                if time.time() - next_stamp < 3600:
                                    found_all = True
                                    self.rate_q.put(next_stamp)

                            #Wait to age out time stamps if exceeded rate limit
                            if (self.rate_q.qsize()>self.rate_limit):
                                next_stamp = self.rate_q.get()
                                remaining_time = 3600 - (time.time() - next_stamp)
                                time.sleep(remaining_time)
                        self.rate_q.put(time.time()+60)
                        try:
                            rsp = fapi.photos_search(api_key=self.flickrAPIkeys[currentKey], ispublic="1", media="photos",
                                                     per_page=str(self.resultsPerPage), page=str(pagenum),
                                                     sort="interestingness-desc", text=query_string,
                                                     extras="tags, original_format, license, geo, date_taken, date_upload, o_dims, views, description",
                                                     min_upload_date=str(mintime),
                                                     max_upload_date=str(maxtime))
                            fapi.testFailure(rsp)
                        except KeyboardInterrupt:
                            print('Keyboard exception while querying for images, exiting\n')
                            raise
                        except (IOError, SSLError) as e:
                            print(('Error on Flickr photo request:{}\n'.format(e.strerror)))
                        except FlickrExpatError as e:
                            print(('Exception encountered while querying for images: {}\n'.format(e.message)))
                            print(('{}: {} to {} page {}\n'.format(query_string, mintime, maxtime, pagenum)))
                            print((e.xmlstr))

                            #I've identified two possible causes of this error: (1)Bad Gateway and (2)bad unicode characters in xml
                            time.sleep(5) #Waiting is best cure for bad gateway
                            pagenum = pagenum + 1 #Skipping to next page is best cure for bad character

                            #Just in case it has some connection to the rate limit, change the key
                            #Randomly choose flickrAPIkeys and flickrAPIsecrets
                            currentKey = int(math.floor(random.random()*len(self.flickrAPIkeys)))
                            # make a new FlickrAPI instance
                            fapi = FlickrAPI(self.flickrAPIkeys[currentKey], self.flickrAPIsecrets[currentKey])

                            self.flickrerrors += 1
                            if self.flickrerrors > 5:
                                print(("Too many Flickr Expat Errors in {}: Exiting".format(self.category)))
                                exit(1)
                        except Exception as e:
                            print((sys.exc_info()[0]))
                            print('Exception encountered while querying for images\n')
                        else:
                            # Process results
                            if getattr(rsp, 'photos', None):
                                if getattr(rsp.photos[0], 'photo', None):
                                    random.shuffle(rsp.photos[0].photo)
                                    for k in range(0, min(self.downloadsPerQuery, len(rsp.photos[0].photo))):
                                        b = rsp.photos[0].photo[k]
                                        if not self.isDuplicateImage(b):
                                            isDownloadable, url = self.get_url(b, fapi, "Medium 640")
                                            if isDownloadable:
                                                b["url"] = url
                                                self.queue.put((b, dirNumName))
                                    print('Waiting threads')
                                    self.queue.join()
                                    while not self.out_queue.empty():
                                        newImages.append(self.out_queue.get())
                                    print((len(newImages), ' downloaded images'))
                            pagenum = pagenum + 1  #this is in the else exception block.  It won't increment for a failure.
                            num_visit_pages = min(4, int(rsp.photos[0]['pages']))
                            # End While of Pages
                # BEGIN: PROCESS DOWNLOADED IMAGES
                self.updateImageNamesIndex(newImages)
            else:
                if command == "exit":
                    self.do_exit = True
                    print(("Wait for safe exit {}".format(self.category)))

        print('End')
        self.cfg.log(self.homeDir, "CRAWLER STOPPED")

