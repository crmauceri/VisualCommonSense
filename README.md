# VisualCommonSense
Extract visual common sense knowledge from Flickr to expand ConceptNet and Freebase
This code was written by me (Cecilia Mauceri) as part of my Master's thesis "Expanding Commonsense Knowledge Bases by Learning from Image Tags". More details of the project are available in the thesis (mauceri2_thesis.pdf)

This code was tested with Python 2.7.8 (Anaconda) and Matlab R2014b on a Windows 8, 64 bit machine

If any functions appear to be missing, check in backup.zip. I tried to clean up the code base for this distribution and may have accidentally removed something.

*********************************** WARNING ***********************************************
* Freebase is depreciated (since June 31, 2015), so parts of this code may no longer work *
*******************************************************************************************

---- External resources ----

Install the following python modules before running the code

nltk
numpy
scipy
requests

Also download the GloVe vector representations from
http://nlp.stanford.edu/projects/glove/

---- Configuration ---

Several lines will need to be changed in the config.txt file

Set sysdir to be the directory where you would like to save the output. 

The next step is to register for API keys for Flickr and Freebase. Insert these keys in the config.txt file on the appropriate lines.

---- Run order ----

1.  python run_download_multiple_query.py <name_of_category or file_path> <max_number_of_images>
    This file downloads the image metadata from Flickr.

    Parameters:
    <name_of_category> is a string containing a Freebase category, such as, /biology/domesticated_animal. 
    <max_number_of_images> is the maximum number of images to download for each search term
    
    A list of the search terms will be saved in  <sysdir>/categories/<name_of_category>.txt
    The image meta data will be saved in <sysdir>/data/texts/

2.  python run_extract_semantics_multiple_query.py <parameters>
    This file extracts the tags from the image metadata, establishes a vocabulary, and counts co-occurence frequency

    Parameters/Flags:
    -q name for the Freebase category. Replace slashes with underscores, e.g. /biology/domesticated_animal becomes biology_domesticated_animal 
    -n minimum number of unique image owners to be included in vocabulary 
    -k Save path for structured knowledge 
    -s Save path for output statistics 
    -r Root directory for image data, usually same as <sysdir>/data/

3. In Matlab: run_analysis( root, search_description, save_description, do_skip, do_approx)
   This file creates the data structures for the cooccurence and vocabulary representations and cleans up the relationships

4. In Matlab: run_

---- Credits ----

The code for this project has built on several previous efforts. The Flickr download scripts were originially written by Tamara Berg, extended heavily by James Hays, modified by Juan C. Caicedo, and shoehorned for new application by Cecilia Mauceri. The semantics extraction code builds on the Iconic Image project by Hongtao Huang and Yunchao Gong.

I also made use of three open source Matlab files from Matlab Central
prec_rec.m     http://www.mathworks.com/matlabcentral/fileexchange/21528-precision-recall-and-roc-curves/content/prec_rec/prec_rec.m
cell2csv.m     http://www.mathworks.com/matlabcentral/fileexchange/7601-cell2csv
JSON.m         http://www.mathworks.com/matlabcentral/fileexchange/42236-parse-json-text/content/JSON.m
