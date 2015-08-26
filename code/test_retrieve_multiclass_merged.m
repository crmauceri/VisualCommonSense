root = 'E:\rooms\data\'; % 'E:\data\Iconic\data'; %  
model = 'glove.42B.300d'; %'glove.twitter.27B.200d', 'glove.6B.300d', 'vectors.840B.300d'
search_description = 'rooms_200_extended'; % 'biology_domesticated_animal_500_extended'; % 'combined_animal_rooms';   % 'cat', 'chair', 'kitchen', 'shoe' 'biology_domesticated_animal' 'fashion_garment'

use_threshold = false;
use_merged_relationships = true; 
use_feature_difference = true; 
min_num_images = 100;
min_images_unlabeled = 100;
min_num_edges = 50;
use_approx = false;
save_str = 'difference';

retrieve_relationship_label(root, search_description, model, use_threshold, ...
    use_merged_relationships, use_feature_difference, min_num_images, min_images_unlabeled, ...
    min_num_edges, use_approx, save_str);        
        