from amr_parser import AMRSemParser

amr = AMRSemParser()
amr.load_cache()

cache_sentences = list(amr.cache.keys())

with open('twc_example_sentences.txt', 'w') as f:
    for cache_sentence in cache_sentences:
        f.write(cache_sentence + '\n')
