function [data, header, concepts] = conceptAnalysis(stats_root, search_description, do_skip)

    %search_description = gen_search_description( query, num_concepts, concept_type );
    savedir = sprintf('%s/Flickr_concepts/', stats_root);
    savepath = sprintf('%s/%s_analysis.mat', savedir, search_description);
    
    if (~do_skip || ~(exist(savepath, 'file') == 2))
        concepts = load(sprintf('%s/concepts/%s_owner_per_concept.mat', stats_root, search_description));
        concepts = cellstr(concepts.concepts);
                
        pos = load(sprintf('%s/structure/%s_pos.mat', stats_root, search_description));
        data = pos.pos_wordnet;
        clear pos;

        language = load(sprintf('%s/structure/%s_languages.mat', stats_root, search_description));
        language_iso = cellstr(language.language_ISO);
        data = [data language.is_language];  
        clear language;

        object = load(sprintf('%s/structure/%s_isImageNet_feature.mat', stats_root, search_description));
        data = [data object.isImageNet']; 
        clear object;

        scene = load(sprintf('%s/structure/%s_SUN_feature.mat', stats_root, search_description));
        data = [data scene.SUN']; 
        clear scene;

        location = load(sprintf('%s/structure/%s_locations.mat', stats_root, search_description));
        population = load(sprintf('%s/structure/%s_population.mat', stats_root, search_description));
        proper_nouns = load(sprintf('%s/structure/%s_proper_nouns.mat', stats_root, search_description));
        lemmas = load(sprintf('%s/structure/%s_lemma_mask.mat', stats_root, search_description));
        
        location_mask = (location.location_mask==1 & (proper_nouns.common_nouns==0 | population.population_mask>30000));
        data = [data (location_mask)']; 
        clear location;
        
        data = [data (proper_nouns.proper_nouns)' (proper_nouns.common_nouns)']; 
        clear proper_nouns;
        
        load(sprintf('%s/filter_lists/colors.mat', stats_root));
        color_mask = ismember(concepts, lower(color_vocab));
        data = [data color_mask];
        clear color_vocab;
        clear color_mask;
        
        data = [data sum(lemmas.lemma_mask, 2)];
        clear lemmas;        
                
        alias = load(sprintf('%s/structure/%s_aliases.mat', stats_root, search_description));
        data = [data sum(tril(alias.aliases_mask, -1), 2)];
        clear alias;
        
        header = [{'pos_N', 'pos_V', 'pos_A', 'pos_v', 'pos_C', ...
            'pos_P', 'pos_!', 'pos_r', 'pos_D', 'pos_I', 'pos_o'}, ...
            language_iso', {'isObject', 'isScene', 'isLocation', ...
            'isProperNoun', 'isCommonNoun', 'isColor', ...
            'isLemma', 'isAlias'}];

        %% Save data
        if ~exist(savedir)
            mkdir(savedir);
        end
        save(savepath, 'data', 'header', 'concepts', 'language_iso');
    else
        load(savepath);
    end

end