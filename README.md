# VisualCommonSense
Extract visual common sense knowledge from Flickr to expand ConceptNet and Freebase

This code was tested with Python 2.7.8 (Anaconda) and Matlab 2014b on a Windows 8, 64 bit machine

---- Modules required -----
Install the following modules before running the code

nltk
numpy
scipy
requests

---- Configuration ---
Several lines will need to be changed in the config.txt file

Set sysdir to be the directory where you would like to save the output. 

The next step is to register for API keys for Flickr and Freebase. Insert these keys in the config.txt file on the appropriate lines.

---- Run order ----

1.  run_download_multiple_query.py <name_of_category or file_path> <max_number_of_images>
    This file downloads the image metadata from Flickr.

    Parameters:
    <name_of_category> is a string containing a Freebase category, such as, /biology/domesticated_animal. 
    <max_number_of_images> is the maximum number of images to download for each search term
    
    A list of the search terms will be saved in  <sysdir>/categories/<name_of_category>.txt
    The image meta data will be saved in <sysdir>/data/texts/

2.  run_extract_semantics_multiple_query.py <parameters>
    This file extracts the tags from the image metadata, establishes a vocabulary, and counts co-occurence frequency

    Parameters/Flags:
    -q name for the Freebase category. Replace slashes with underscores, e.g. /biology/domesticated_animal becomes biology_domesticated_animal 
    -n minimum number of unique image owners to be included in vocabulary 
    -k Save path for structured knowledge 
    -s Save path for output statistics 
    -r Root directory for image data, usually same as <sysdir>/data/



---- Credits ----

The code for this project has built on several previous efforts. The Flickr download scripts were originially written by Tamara Berg, extended heavily by James Hays, modified by Juan C. Caicedo, and shoehorned for new application by Cecilia Mauceri. The semantics extraction code builds on the Iconic Image project by Hongtao Huang and Yunchao Gong.