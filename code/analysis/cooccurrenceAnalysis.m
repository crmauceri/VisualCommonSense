function [data, dataDims] = cooccurrenceAnalysis(stats_root, search_description, post_fix, method, do_skip)
    %% Calculates various statistical metrics of cooccurrence of flickr tag pairs
    % method = 'images', 'ngrams', OR 'owners'
    % The output is saved as a matlab mat file.
    % The 3rd dimension slices represent
    %   1.) Is in Freebase
    %   2.) Is in ConceptNet
    %   3.) Is a lemma
    %   4.) Is a synonym
    %   5.) Number of images or owners depending on method
    %   6.) Probability
    %   7.) PMI
    %   8.) minimum conditional probability
    %   9.) mean conditional probability
    %   10.) conditional probability

    variable_def
    %search_description = gen_search_description( query, num_concepts, concept_type );
    savedir = sprintf('%s/Flickr_cooccurrence/', stats_root);
    savepath = sprintf('%s/%s_pattern_analysis_%s.mat', savedir, search_description, method);
        
    if (~do_skip || ~(exist(savepath, 'file') == 2))
        freebase = load(sprintf('%s/structure/%s_Freebase_adjacency.mat', stats_root, search_description));
        %freebase = freebase.adjacent;
        actual_num_tags = freebase.shape(1);
        data = zeros(actual_num_tags, actual_num_tags, MAX_DIM);
        data(:,:,IS_FREEBASE) = reshape(sum(freebase.adjacency, 2), freebase.shape(1:2));
        clear freebase;

        conceptNet = load(sprintf('%s/structure/%s_ConceptNet_adjacency.mat', stats_root, search_description));
        %conceptNet = conceptNet.adjacent;
        data(:,:,IS_CONCEPTNET) = reshape(sum(conceptNet.adjacency, 2), conceptNet.shape(1:2));
        clear conceptNet;

        synonyms = load(sprintf('%s/structure/%s_synonym_mask.mat', stats_root, search_description));
        data(:,:,IS_SYNONYM) = synonyms.synonym_mask;
        clear synonyms;
        
        lemmas = load(sprintf('%s/structure/%s_lemma_mask.mat', stats_root, search_description));
        data(:,:,IS_LEMMA) = lemmas.lemma_mask;
        clear lemmas;        
                
        alias = load(sprintf('%s/structure/%s_aliases.mat', stats_root, search_description));
        data(:,:,IS_ALIAS) = alias.aliases_mask;
        clear alias;
        
        language = load(sprintf('%s/structure/%s_languages.mat', stats_root, search_description));
        data(:,:,IS_TRANSLATION) = language.is_translation;
        clear language;

        merge_mask = data(:,:,IS_LEMMA)' | data(:,:,IS_ALIAS);
        flickr = load(sprintf('%s/structure/%s_%s.mat', stats_root, search_description, post_fix));        
        switch method
           case 'images'
               display('WARNING missing variables');
               diagonal = diag(diag(flickr.comatrix_images));
               flickr.comatrix_images = flickr.comatrix_images + flickr.comatrix_images' - diagonal;
               [total_vocab_count, total_cooccurrence] = add_plurals_to_singular(merge_mask, flickr.comatrix_images, flickr.vocab_image_count);
               data(:,:,NUM_IMAGES) = total_cooccurrence;
               
               %TODO fix probability bug from merging lemmas
               total_cooccurrence = flickr.comatrix_images;
               total_vocab_count = flickr.vocab_image_count;
               
               total_instances = double(flickr.total_images);

           case 'owners'
               diagonal = diag(diag(flickr.comatrix_images));
               flickr.comatrix_images = flickr.comatrix_images + flickr.comatrix_images' - diagonal;
               [~, total_cooccurrence_images] = add_plurals_to_singular(merge_mask, flickr.comatrix_images, zeros(size(flickr.vocab_owner_count)));
               %total_cooccurrence_images = flickr.comatrix_images;
               data(:,:,NUM_IMAGES) = total_cooccurrence_images;
        
               diagonal = diag(diag(flickr.comatrix_owners));
               flickr.comatrix_owners = flickr.comatrix_owners + flickr.comatrix_owners' - diagonal;
               
               [total_vocab_count, total_cooccurrence] = add_plurals_to_singular(merge_mask, flickr.comatrix_owners, flickr.vocab_owner_count);
               data(:,:,NUM_OWNERS) = total_cooccurrence;
               
               %TODO fix probability bug from merging lemmas
               total_cooccurrence = flickr.comatrix_owners;
               total_vocab_count = flickr.vocab_owner_count;
              
               total_instances = double(flickr.total_owners);
        end
        %% Normalized Cooccurance 
        data(:,:,PROB) = total_cooccurrence/total_instances;
        
        %% Pointwise Mutual Information
        data(:,:,N_PMI) = normalized_pointwise(total_cooccurrence, total_vocab_count, total_instances);
        data(:,:,PMI) = pointwise_bound(total_cooccurrence, total_vocab_count, total_instances);

        %% Conditional Probability
        data(:,:,MIN_COND_PROB) = conditional(total_cooccurrence, total_vocab_count, total_instances, 'min');
        data(:,:,MEAN_COND_PROB) = conditional(total_cooccurrence, total_vocab_count, total_instances, 'mean');
        data(:,:,COND_PROB) = conditional(total_cooccurrence, total_vocab_count, total_instances, 'direct');

        %% Use compressed format
        dataDims = size(data);
        data = sparse(reshape(data, actual_num_tags*actual_num_tags, []));

        %% Save data       
        if ~exist(savedir)
            mkdir(savedir)
        end
        save(savepath, 'data', 'dataDims', '-v7.3');
    
    else
        load(savepath);
    end

end