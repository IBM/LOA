import os
import re

from amr_parser import get_verbnet_preds_from_obslist


def protected_tokenizer(sentence_string, simple=False):

    if simple:
        # simplest possible tokenizer
        # split by these symbols
        sep_re = re.compile(r'[\.,;:?!"\' \(\)\[\]\{\}]')
        return simple_tokenizer(sentence_string, sep_re)
    else:
        # imitates JAMR (97% sentece acc on AMR2.0)
        # split by these symbols
        # TODO: Do we really need to split by - ?
        sep_re = re.compile(r'[/~\*%\.,;:?!"\' \(\)\[\]\{\}-]')
        return jamr_like_tokenizer(sentence_string, sep_re)


def simple_tokenizer(sentence_string, separator_re):

    tokens = []
    positions = []
    start = 0
    for point in separator_re.finditer(sentence_string):

        end = point.start()
        token = sentence_string[start:end]
        separator = sentence_string[end:point.end()]

        # Add token if not empty
        if token.strip():
            tokens.append(token)
            positions.append((start, end))

        # Add separator
        if separator.strip():
            tokens.append(separator)
            positions.append((end, point.end()))

        # move cursor
        start = point.end()

    # Termination
    end = len(sentence_string)
    if start < end:
        token = sentence_string[start:end]
        if token.strip():
            tokens.append(token)
            positions.append((start, end))

    return tokens, positions


def jamr_like_tokenizer(sentence_string, sep_re):

    # quote normalization
    sentence_string = sentence_string.replace('``', '"')
    sentence_string = sentence_string.replace("''", '"')
    sentence_string = sentence_string.replace("“", '"')

    # currency normalization
    #sentence_string = sentence_string.replace("£", 'GBP')

    # Do not split these strings
    protected_re = re.compile("|".join([
        # URLs (this conflicts with many other cases, we should normalize URLs
        # a priri both on text and AMR)
        # r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*',
        #
        r'[0-9][0-9,\.:/-]+[0-9]',         # quantities, time, dates
        r'^[0-9][\.](?!\w)',               # enumerate
        r'\b[A-Za-z][\.](?!\w)',           # itemize
        r'\b([A-Z]\.)+[A-Z]?',             # acronym with periods (e.g. U.S.)
        r'!+|\?+|-+|\.+',                  # emphatic
        r'etc\.|i\.e\.|e\.g\.|v\.s\.|p\.s\.|ex\.',     # latin abbreviations
        r'\b[Nn]o\.|\bUS\$|\b[Mm]r\.',     # ...
        r'\b[Mm]s\.|\bSt\.|\bsr\.|a\.m\.',  # other abbreviations
        r':\)|:\(',                        # basic emoticons
        # contractions
        r'[A-Za-z]+\'[A-Za-z]{3,}',        # quotes inside words
        r'n\'t(?!\w)',                     # negative contraction (needed?)
        r'\'m(?!\w)',                      # other contractions
        r'\'ve(?!\w)',                     # other contractions
        r'\'ll(?!\w)',                     # other contractions
        r'\'d(?!\w)',                      # other contractions
        # r'\'t(?!\w)'                      # other contractions
        r'\'re(?!\w)',                     # other contractions
        r'\'s(?!\w)',                      # saxon genitive
        #
        r'<<|>>',                          # weird symbols
        #
        r'Al-[a-zA-z]+|al-[a-zA-z]+',      # Arabic article
        # months
        r'Jan\.|Feb\.|Mar\.|Apr\.|Jun\.|Jul\.|Aug\.|Sep\.|Oct\.|Nov\.|Dec\.'
    ]))

    # iterate over protected sequences, tokenize unprotected and append
    # protected strings
    tokens = []
    positions = []
    start = 0
    for point in protected_re.finditer(sentence_string):

        # extract preceeding and protected strings
        end = point.start()
        preceeding_str = sentence_string[start:end]
        protected_str = sentence_string[end:point.end()]

        if preceeding_str:
            # tokenize preceeding string keep protected string as is
            for token, (start2, end2) in zip(
                *simple_tokenizer(preceeding_str, sep_re)
            ):
                tokens.append(token)
                positions.append((start + start2, start + end2))
        tokens.append(protected_str)
        positions.append((end, point.end()))

        # move cursor
        start = point.end()

    # Termination
    end = len(sentence_string)
    if start < end:
        ending_str = sentence_string[start:end]
        if ending_str.strip():
            for token, (start2, end2) in zip(
                *simple_tokenizer(ending_str, sep_re)
            ):
                tokens.append(token)
                positions.append((start + start2, start + end2))

    return tokens, positions


amr_server_ip = os.getenv('LOA_AMR_SERVER_IP', 'localhost')
amr_server_port = int(os.getenv('LOA_AMR_SERVER_PORT', '0'))


obs = """
-= Laundry Room =-
You find yourself in a laundry room. An usual one.
Okay, just remember what you're here to do,
and everything will go great.
You make out a washing machine. Empty!
What kind of nightmare TextWorld is this?
You can make out an opened laundry basket. Empty!
What kind of nightmare TextWorld is this?
You make out a clothes drier. The clothes drier is empty!
This is the worst thing that could possibly happen, ever!
You scan the room, seeing a suspended shelf.
Unfortunately, there isn't a thing on it. You see a work table.
On the work table you can see a pair of dirty gray underpants.
You make out a bench.
Looks like someone's already been here and taken everything off it,
though. Aw, here you were,
all excited for there to be things on it!'
"""

# tokens, positions = protected_tokenizer(obs)
#
# print(tokens)
# print(positions)
#
# exit()

all_preds, pred_count_dict, verbnet_facts_logs = \
    get_verbnet_preds_from_obslist(
        # obslist=['You pick up the wet hoodie from the ground. ' +
        #          'You pick up an red apple from the black big table.'],
        # obslist=['You pick up the wet hoodie from the ground and ' +
        #          'you make out coat hanger.'],
        obslist=['you are carrying peanut oil, flour, and sugar.'],
        # obslist=['I see the cake.'],
        # obslist=['I saw her bake the cake.'],
        # obslist=[
        #     '-= Backyard =- ' +
        #     'I just think it\'s great you\'ve just entered a backyard. ' +
        #     'I guess you better just go and list everything you see here. ' +
        #     'You can make out a BBQ. The BBQ is recent. ' +
        #     'But the thing hasn\'t got anything on it. ' +
        #     'What you think everything in TextWorld should have stuff? ' +
        #     'You make out a clothesline. The clothesline is typical. ' +
        #     'But the thing is empty, unfortunately. ' +
        #     'Aw, here you were, all excited for there to be things on it! ' +
        #     'As if things weren\'t amazing enough already, ' +
        #     'you can even see a patio chair. The patio chair is stylish. ' +
        #     'However, the patio chair, like an empty patio chair, ' +
        #     'has nothing on it. You make out a patio table. ' +
        #     'The patio table is stylish. But there isn\'t a thing on it. ' +
        #     'You bend down to tie your shoe. When you stand up, ' +
        #     'you notice a workbench. ' +
        #     'On the workbench you see a wet cardigan. ' +
        #     'There is an open screen door leading west.'
        # ],
        # obslist=[
        #     """
        #     -= Laundry Room =-
        #     You find yourself in a laundry room. An usual one.
        #     Okay, just remember what you're here to do,
        #     and everything will go great.
        #     You make out a washing machine. Empty!
        #     What kind of nightmare TextWorld is this?
        #     You can make out an opened laundry basket. Empty!
        #     What kind of nightmare TextWorld is this?
        #     You make out a clothes drier. The clothes drier is empty!
        #     This is the worst thing that could possibly happen, ever!
        #     You scan the room, seeing a suspended shelf.
        #     Unfortunately, there isn't a thing on it. You see a work table.
        #     On the work table you can see a pair of dirty gray underpants.
        #     You make out a bench.
        #     Looks like someone's already been here and taken everything off it,
        #     though. Aw, here you were,
        #     all excited for there to be things on it!'
        #     """
        # ],
        amr_server_ip=amr_server_ip,
        amr_server_port=amr_server_port,
        mincount=0, verbose=True,
        sem_parser_mode='propbank',
    )

print(all_preds)
print(pred_count_dict)
print(verbnet_facts_logs)
