import ast
import itertools
import os
import pickle
import re

import requests
import spacy
from nltk.tokenize import sent_tokenize
from tqdm import tqdm


def get_node_id(amr):
    lines = amr.split('\n')
    node_alignments = dict()
    surface_form_nodes = []
    surface_form_indices = [-1, -1]

    for line in lines:
        if line.startswith('# ::tok '):
            line = line.replace('# ::tok ', '')
            tokens = line.split(' ')
        if line.startswith('# ::node'):
            splits = line.split()
            if len(splits) < 4:
                continue
            elif len(splits) == 4:
                node_id = splits[2]
                node_label = splits[3]
                surface_form = node_label
                node_alignments[node_id] = [node_label, surface_form]
            else:
                span_splits = splits[4].split('-')
                node_id = splits[2]
                node_label = splits[3]
                if int(span_splits[0]) >= surface_form_indices[0] \
                        and int(span_splits[1]) <= surface_form_indices[1]:
                    surface_form_nodes.append((node_id, node_label))
                surface_form = \
                    ' '.join(tokens[int(span_splits[0]):int(span_splits[1])])
                node_alignments[node_id] = [node_label, surface_form]
        if line.startswith('# ::short'):
            splits = line.split('\t')
            align_amr_node_id = ast.literal_eval(splits[1])

    return node_alignments, align_amr_node_id


def get_verbnet_preds_from_obslist(obslist,
                                   amr_server_ip='localhost',
                                   amr_server_port=None,
                                   mincount=0, verbose=False,
                                   sem_parser_mode='both'):
    rest_amr = AMRSemParser(amr_server_ip=amr_server_ip,
                            amr_server_port=amr_server_port)
    all_preds = []
    verbnet_facts_logs = {}
    for obs_text in tqdm(obslist):
        verbnet_facts, arity = \
            rest_amr.obs2facts(obs_text,
                               verbose=verbose,
                               mode=sem_parser_mode,
                               force_single_arity=False)
        verbnet_facts_logs[obs_text] = verbnet_facts
        all_preds += list(verbnet_facts.keys())

    rest_amr.save_cache()

    all_preds_set = list(set(all_preds))

    pred_count_dict = {k: all_preds.count(k) for k in all_preds_set}
    all_preds = [k for k, v in pred_count_dict.items() if v > mincount]

    if verbose:
        print('Found {} verbnet preds'.format(len(all_preds)))
        print('Predicates are: ', all_preds)

    return all_preds, pred_count_dict, verbnet_facts_logs


def get_formatted_obs_text(infos):
    obs = infos['description']
    sent_part1 = infos['inventory'].split(':\n')[0]
    sent_part2 = ', '.join(infos['inventory'].split(':\n')[1:])[2:]
    sent = obs.replace('\n', ' ') + ' ' + sent_part1 + ' ' + sent_part2
    return sent


def remove_nextline_space(in_text):
    return re.sub(' +', ' ', in_text)


def detect_joined_noun_phrases(sent, join_token='of', self_assign=False):
    words = sent.split()
    token_dict = {}
    for k, token in enumerate(words):
        if k >= 1 and k < len(words) - 1 and token == join_token:
            full_ent = ' '.join(words[k - 1:k + 2])
            token_dict[words[k - 1]] = full_ent
            token_dict[words[k + 1]] = full_ent
            if self_assign:
                token_dict[full_ent] = full_ent
    return token_dict


def remove_article(s):
    article_list = ['a', 'an', 'the']
    ws = [x for x in s.split() if x not in article_list]
    return ' '.join(ws)


class AMRSemParser:
    def __init__(self,
                 amr_server_ip='localhost',
                 amr_server_port=None,
                 use_amr_cal_str=False,
                 cache_folder='./cache/'):
        self.use_amr_cal_str = use_amr_cal_str
        if amr_server_port == 0 or amr_server_port is None:
            print('AMR is cache only mode')
            self.endpoint = None
        else:
            self.endpoint = \
                'http://%s:%d/verbnet_semantics' % \
                (amr_server_ip, amr_server_port)
        if not os.path.exists(cache_folder):
            os.mkdir(cache_folder)
        self.cache_file = cache_folder + 'amr_cache.pkl'
        self.json_key = 'amr_parse'

        self.nlp = spacy.load('en_core_web_sm')
        self.cache = {}
        self.load_cache()

    def text2amr(self, text):
        full_ret = {self.json_key: []}
        for sent in sent_tokenize(text):
            if sent in self.cache:
                ret = self.cache[sent]
                full_ret[self.json_key].append(ret)
            else:
                if self.endpoint is None:
                    raise Exception('Need the AMR server for "' + sent + '"')
                r = requests.get(self.endpoint,
                                 params={'text': sent, 'use_coreference': 0})
                ret = r.json()
                self.cache[sent] = ret[self.json_key][0]
                full_ret[self.json_key].append(ret[self.json_key][0])

        return full_ret

    def load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'rb') as fp:
                self.cache = pickle.load(fp)
            print('Loaded cache from ', self.cache_file)
        else:
            self.cache = {}

    def save_cache(self):
        with open(self.cache_file, 'wb') as fp:
            pickle.dump(self.cache, fp)
        print('Saved cache to ', self.cache_file)

    def propbank_facts(self, ret,
                       no_use_zero_arg=True,
                       force_single_arity=True,
                       verbose=True, cnt=None):
        amr_text = ret[self.json_key][cnt]['amr']
        if verbose:
            print('Text:')
            print(ret[self.json_key][cnt]['text'])
        amr_cal_text = ret[self.json_key][cnt]['amr_cal']

        node_alignments, align_amr_node_id = get_node_id(amr_text)
        # filter None cases in the keys
        align_amr_node_id = {k: v for k, v in align_amr_node_id.items()
                             if k is not None}
        node_alignments_temp = {int(k): v[1].lower()
                                for k, v in node_alignments.items()
                                if k != 'None'}

        # Disambiguate between different objects
        values = list(node_alignments_temp.values())
        ent_idx = {}
        node_alignments = {}
        for k, v in node_alignments_temp.items():
            count_v = values.count(v)
            if count_v > 1:
                if v in ent_idx:
                    ent_idx[v] -= 1
                else:
                    ent_idx[v] = count_v
                node_alignments[k] = v + '_count_' + str(ent_idx[v])
            else:
                node_alignments[k] = v
        node2surface_mapping = {v: node_alignments[k] for k, v in
                                align_amr_node_id.items()}

        if verbose:
            print('AMR Cal Text:')
            for k, v in ret[self.json_key][cnt].items():
                print(k, ':', v)

        pred_values = {}
        for item in amr_cal_text:
            pred_name = item['predicate']

            if '-' in pred_name and 'arg' in pred_name:
                verbnet_frame = \
                    '-'.join(pred_name.split('.')[0].split('-')[:-1])
                is_neg = item['is_negative']
                if is_neg:
                    verbnet_frame = 'not_' + verbnet_frame

                arg_name = node2surface_mapping[item['arguments'][1]]

                arg_no = int(pred_name.split('.')[-1].split('arg')[-1])

                if (arg_no == 0 and no_use_zero_arg) \
                        or verbnet_frame == 'have-mod':
                    continue

                if verbnet_frame not in pred_values:
                    pred_values[verbnet_frame] = dict()
                if item['arguments'][0] not in pred_values[verbnet_frame]:
                    pred_values[verbnet_frame][item['arguments'][0]] = dict()

                if verbose:
                    if arg_no \
                            in pred_values[verbnet_frame][item['arguments'][0]]:
                        print('override (%s, %s, %d): %s -> %s' %
                              (verbnet_frame, item['arguments'][0], arg_no,
                               pred_values[verbnet_frame]
                               [item['arguments'][0]][arg_no], arg_name))

                pred_values[verbnet_frame][item['arguments'][0]][arg_no] = \
                    arg_name

        facts = {}
        arity = {}
        for verb, pred_value in pred_values.items():
            for _, value in pred_value.items():
                pred_values_list = \
                    [v[1] for v in sorted(value.items(), key=lambda i: i[0])]
                if len(pred_values_list) == 1 or \
                        (len(pred_values_list) > 1 and not force_single_arity):
                    if verb not in facts:
                        facts[verb] = []
                    facts[verb].append(pred_values_list)
                    if verb in arity and arity[verb] != len(pred_values_list):
                        arity[verb] = max(arity[verb], len(pred_values_list))
                    else:
                        arity[verb] = len(pred_values_list)

        return facts, arity

    def verbnet_facts(self, ret,
                      no_use_zero_arg=True,
                      force_single_arity=True,
                      verbose=True, cnt=None):
        facts = {}

        res = ret[self.json_key][cnt]
        amr_text = res['amr']

        node_alignments, align_amr_node_id = get_node_id(amr_text)
        # filter None cases in the keys
        node_alignments_temp = {int(k): v[1].lower() for k, v in
                                node_alignments.items() if k != 'None'}
        # Disambiguate between different objects
        values = list(node_alignments_temp.values())
        ent_idx = {}
        node_alignments = {}
        for k, v in node_alignments_temp.items():
            count_v = values.count(v)
            if count_v > 1:
                if v in ent_idx:
                    ent_idx[v] -= 1
                else:
                    ent_idx[v] = count_v
                node_alignments[k] = v + '_count_' + str(ent_idx[v])
            else:
                node_alignments[k] = v
        node2surface_mapping = {v: node_alignments[k] for k, v in
                                align_amr_node_id.items()}

        if verbose:
            print('##' * 30)
            print('Grounded smt: ', res['grounded_stmt'])
            print('sem_cal_str: ', res['sem_cal_str'])

        for k, v in res['grounded_stmt'].items():
            verb = k.split('.')[0]
            key_desired = [k_in for k_in in v if verb in k_in]

            if len(key_desired) > 0:
                key_desired = key_desired[0]
            else:
                continue

            for item in v[key_desired][0]:
                pred = item['predicate']
                try:
                    val_facts = tuple([node2surface_mapping[x] for x in
                                       item['arguments'][1:]])
                    if pred in facts:
                        facts[pred].append(val_facts)
                    else:
                        facts[pred] = [val_facts]
                except BaseException:
                    pass
        arity = {}
        for k, v in facts.items():
            v_tuple = []
            arity[k] = []
            for item in v:
                arity[k].append(len(item))
                if len(item) > 1:
                    if force_single_arity:
                        v_tuple += item
                    else:
                        v_tuple.append(tuple(item))
                else:
                    v_tuple.append(item[0])
            arity[k] = list(set(arity[k]))
            v = list(set(v_tuple))
            facts[k] = v
        return facts, arity

    def get_all_possible_adj_nouns(self, phrase):
        list_out = []
        phrase_split = phrase.split()
        for k in range(0, len(phrase_split)):
            list_out.append(' '.join(phrase_split[k:]))
        return list_out

    def get_entity_mappings(self, text, filter_quantifiers, quantifer_words,
                            add_self_mapping=False,
                            add_joined_words=True):

        doc = self.nlp(text)
        list_nps = []
        for nphrase in doc.noun_chunks:
            list_nps.append(remove_article(nphrase.text.lower()))
        list_nps = list(set(list_nps))

        list_nps_dict = {}
        list_root_noun = []
        for x in list_nps:
            if filter_quantifiers:
                x = ' '.join([item for item in x.split()
                              if item not in quantifer_words])
            list_root_noun.append(x.split()[-1])
        ent_idx = {}
        for v in list_nps:
            root_noun = v.split()[-1]
            count_v = list_root_noun.count(root_noun)
            if count_v > 1:
                if root_noun in ent_idx:
                    ent_idx[root_noun] -= 1
                else:
                    ent_idx[root_noun] = count_v
                key = root_noun + '_count_' + str(ent_idx[root_noun])
            else:
                key = root_noun
            list_nps_dict[key] = v
            if add_self_mapping:
                list_nps_dict[v] = v
        if add_joined_words:
            joined_words_dict = detect_joined_noun_phrases(text)
            list_nps_dict = {**list_nps_dict, **joined_words_dict}
        return list_nps_dict

    def obs2facts(self, text, no_use_zero_arg=True, force_single_arity=True,
                  mode='both',
                  verbose=False, filter_nps=True, filter_quantifiers=True):

        text = remove_nextline_space(' and '.join(text.split('\n')))
        ret = self.text2amr(text)
        final_facts = {}
        final_arity = {}

        quantifer_words = ['some ', 'many ', 'lot ', 'few ']
        full_list_nps_dict = self.get_entity_mappings(text, filter_quantifiers,
                                                      quantifer_words)

        for cnt in range(len(ret[self.json_key])):
            if mode == 'both':
                propbank_facts, propbank_arity_facts = \
                    self.propbank_facts(ret,
                                        no_use_zero_arg=no_use_zero_arg,
                                        cnt=cnt,
                                        force_single_arity=force_single_arity,
                                        verbose=verbose)
                verbnet_facts, verbnet_arity_facts = \
                    self.verbnet_facts(ret,
                                       no_use_zero_arg=no_use_zero_arg,
                                       cnt=cnt,
                                       force_single_arity=force_single_arity,
                                       verbose=verbose)

                facts = {**verbnet_facts, **propbank_facts}
                arity = {**verbnet_arity_facts, **propbank_arity_facts}
            elif mode == 'verbnet':
                verbnet_facts, verbnet_arity_facts = \
                    self.verbnet_facts(ret,
                                       no_use_zero_arg=no_use_zero_arg,
                                       cnt=cnt,
                                       force_single_arity=force_single_arity,
                                       verbose=verbose)
                facts = verbnet_facts
                arity = verbnet_arity_facts
            elif mode == 'propbank':
                propbank_facts, propbank_arity_facts =\
                    self.propbank_facts(
                        ret,
                        no_use_zero_arg=no_use_zero_arg,
                        cnt=cnt,
                        force_single_arity=force_single_arity,
                        verbose=verbose)
                facts = propbank_facts
                arity = propbank_arity_facts
            elif mode == 'none':
                facts = {}
                arity = {}
            else:
                print('Invalid mode. exitting...')
                return None

            # Add handicap in NER based entity linking
            text_sub = ret[self.json_key][cnt]['text']
            if filter_nps:
                list_nps_dict = self.get_entity_mappings(text_sub,
                                                         filter_quantifiers,
                                                         quantifer_words)
                facts_filtered = {}
                for kk, vv in facts.items():
                    for v in vv:
                        v_filtered = [list_nps_dict[item] for item in v
                                      if item in list_nps_dict]
                        if len(v_filtered) == 0:
                            v_filtered = [full_list_nps_dict[item]
                                          for item in v
                                          if item in full_list_nps_dict]
                        if len(v_filtered) > 0:
                            if kk not in facts_filtered:
                                facts_filtered[kk] = list()
                            facts_filtered[kk].append(v_filtered)
            else:
                facts_filtered = facts

            if verbose:
                print('Text: ', text_sub)
                print('AMR Sem Cal: \n',
                      ret[self.json_key][cnt]['amr_cal_str'])
                print('Facts: \n', facts_filtered)
                print('#' * 50)

            final_facts.update(facts_filtered)
            final_arity.update(arity)

        for k, vs in final_facts.items():
            final_facts[k] = list()
            for v in vs:
                words = list()
                for word in v:
                    words.append(self.get_all_possible_adj_nouns(word))
                for f in itertools.product(*words):
                    final_facts[k].append(list(f))

        return final_facts, final_arity
