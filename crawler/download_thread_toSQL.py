import os, sys, json
import threading
import re
from datetime import datetime
from string import digits
#import database_builder.build_database as db
#from sqlalchemy import create_engine
from nltk.corpus import stopwords

cachedStopWords = stopwords.words("english")
#Preload strptime to prevent thread bug
datetime.strptime("2000-10-10 0:0:0", '%Y-%m-%d %H:%M:%S')

def is_good_word(word):
    if len(word) <= 2:
        return 0
    if word in cachedStopWords:
        return 0
    return 1

def simpleDateFormat(date):
    d = date.split()
    return d[0].replace("-","")

# ######################################
# DOWNLOAD IMAGE THREAD
# ######################################
class DownloadImageThread(threading.Thread):
    def __init__(self, queue, out_queue, outputDirImg, outputDirTxt, query, database):
        threading.Thread.__init__(self)
        #self.engine = create_engine(database)
        self.query = query
        self.queue = queue
        self.out_queue = out_queue
        self.outputDirImg = outputDirImg
        self.outputDirTxt = outputDirTxt
        self.outputFileIdx = 0
        self.remove_digits_dict = str.maketrans("", "", digits)

    def is_valid_photo(self, photo):
        necessary_key = ['id',
                         'secret',
                         'server',
                         'owner',
                         'title',
                         'datetaken',
                         'tags']
        bad_key = []
        is_valid = True
        for key in necessary_key:
            if (key not in photo.attrib):
                is_valid = False
                bad_key.append(key)
        return is_valid

    def escape(self, c):
        if c.group(0) == '\n':
            return '\\n'
        elif c.group(0) == '\v':
            return '\\v'
        elif c.group(0) == '\t':
            return '\\t'
        elif c.group(0) == '\r':
            return '\\r'
        elif c.group(0) == '\f':
            return '\\f'
        elif c.group(0) == '\b':
            return '\\b'
        elif c.group(0) == '\\':
            return '\\\\'
        else:
            return '\\' + c.group(0)

    def clean_photo(self, photo, imagefile):
        photo['imagepath'] = imagefile

        title = str(photo['title'].encode('ascii', 'replace'))
        photo['title'] = re.sub('[\\\\\"\v\t\r\n\f\b]', self.escape, title)

        tags = str(photo['tags'].encode('ascii', 'replace'))

        tags = tags.translate(self.remove_digits_dict).split(' ')
        tags = [t.lower() for t in tags if is_good_word(t)]

        if getattr(photo, 'description', None):
            description = str(photo.description[0].elementText.encode('ascii', 'replace')).strip()
            photo['description'] = re.sub('[\\\\\"\v\t\r\n\f\b]', self.escape, description)

        # try:
        #     photo['datetaken'] = datetime.strptime(photo['datetaken'], '%Y-%m-%d %H:%M:%S').date()
        # except:
        #      print sys.exc_info()[0]
        #      print ('Error encountered on parsing data\n')
        #      photo['datetaken'] = datetime.now().date()

        photo = photo.attrib

        #Remove tags key because will be stored seperately in database
        del photo['tags']
        return photo, tags

    def run(self):
        # conn = self.engine.connect()
        # image_inserter = db.image.insert().prefix_with("IGNORE")
        # concept_inserter = db.concept.insert().prefix_with("IGNORE")
        # image_query_inserter = db.image_query.insert().prefix_with("IGNORE")
        # image_concept_inserter = db.image_concept.insert().prefix_with("IGNORE")

        processed_photo_list = []
        photo_query_list = []
        tag_list = set([])
        tag_image_list = []
        while True:
            photo_XML, dirNumName = self.queue.get()
            if photo_XML != None and self.is_valid_photo(photo_XML):
                date = simpleDateFormat(photo_XML['datetaken'])
                file = photo_XML['server'] + '_' + photo_XML['id'] + '_' + photo_XML['secret'] + '_' + photo_XML['owner'] + 'D' + date
                imagepath = self.outputDirImg + dirNumName + file + '.jpg'
                photo_XML, tags = self.clean_photo(photo_XML, imagepath)
                if len(tags) > 0:
                    processed_photo_list.append(photo_XML)
                    photo_query_list.append({'query_str': self.query, 'image_key': photo_XML['url']})
                    tag_list.update(tags)
                    tag_image_list += [{'concept_key': t, 'image_key': photo_XML['url'], 'isTag': True} for t in tags]
                    self.out_queue.put(file +'.jpg')

                    if len(processed_photo_list) > 100:
                        save_path = self.outputDirTxt + dirNumName + str(self.outputFileIdx).rjust(5, '0') + self.name + '.txt'
                        while(os.path.exists(save_path)):
                            self.outputFileIdx += 1
                            save_path = self.outputDirTxt + dirNumName + str(self.outputFileIdx).rjust(5, '0') + self.name + '.txt'

                        with open(save_path, 'w') as f:
                            json.dump({'processed_photo_list': processed_photo_list,
                                       'photo_query_list':photo_query_list,
                                       'tag_list':list(tag_list),
                                       'tag_image_list':tag_image_list}, f)
                        self.outputFileIdx += 1
                        processed_photo_list = []
                        photo_query_list= []
                        tag_list=set([])
                        tag_image_list=[]
                        # if(self.try_insert(conn, image_inserter, processed_photo_list)):
                        #     processed_photo_list = []
                        # if(self.try_insert(conn, image_query_inserter, photo_query_list)):
                        #     photo_query_list = []
                        # if(self.try_insert(conn, concept_inserter, [{'concept_str': t} for t in tag_list])):
                        #     tag_list = set([])
                        # if(self.try_insert(conn, image_concept_inserter, tag_image_list)):
                        #     tag_image_list = []

            self.queue.task_done()

    def try_insert(self, conn, query, insert_list):
        inserted = False
        num_try = 0
        while not inserted and num_try<3:
            try:
                conn.execute(query, insert_list)
            except:
                (sys.exc_info()[0])
                print('Deadlock encountered on insert\n')
                num_try += 1
            else:
                inserted = True
        return inserted

    def setValidTags(self, validTags):
        self.validTags = validTags

